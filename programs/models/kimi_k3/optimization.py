"""Kimi-K3 text-model inference — graph splitting and placement."""

from collections import defaultdict, deque
from fractions import Fraction

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import (
    AllGather, Broadcast, CommKernel, Gather, Reduce, Scatter, Send,
)
from rooflang.language.kernels.forward import (
    KimiK3DeltaAttnCpMerge, KimiK3DeltaAttnCpSummary, Nop,
)
from rooflang.language.optimization.comm import optimize_comms
from rooflang.language.optimization.split import (
    batch_split, context_split_decode, context_split_prefill, general_dup,
)
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor

from rooflang.programs.models.kimi_k3.config import (
    H, KDA_CHUNK, KDA_CONV, KDA_HD,
    N_EXPERTS, TOPK,
)


_PREFILL_PARALLEL_FIELDS = (
    "bridge", "block_in_fan", "block_out_fan", "pre_attn_res",
    "block_append", "attn_norm", "attn_fan", "wq_a", "q_norm",
    "wq_b", "wkv", "kv_norm", "kda_wq", "kda_wk", "kda_wv",
    "kda_f_a", "kda_f_b", "kda_beta", "output_gate",
    "kv_cache_quant", "kv_persist_fan", "sa", "sa_fan",
    "attn_out_norm", "attn_gate", "state_store", "wo", "attn_add",
    "ffn_bridge", "mlp_res", "ffn_norm", "ffn_fan", "dense_up",
    "dense_down", "gate", "routed_down", "dispatch", "combine",
    "routed_norm", "routed_up", "sw_up", "sw_down", "moe_add",
    "ffn_add",
)

_DECODE_MLA_DP_ONLY_FIELDS = (
    "bridge", "block_in_fan", "block_out_fan", "pre_attn_res",
    "block_append", "attn_norm", "attn_fan", "wq_a", "q_norm",
    "wq_b", "wkv", "kv_norm", "kv_sink", "output_gate",
)

_DECODE_MLA_REPLICATED_FIELDS = ()

_DECODE_MLA_SHARDED_FIELDS = (
    "attn_gate", "wo", "attn_add", "ffn_bridge", "mlp_res",
    "ffn_norm", "ffn_fan", "dense_up", "dense_down", "gate",
    "routed_down", "dispatch", "combine", "routed_norm", "routed_up",
    "sw_up", "sw_down", "moe_add", "ffn_add",
)

_DECODE_KDA_BATCH_FIELDS = (
    "bridge", "block_in_fan", "block_out_fan", "pre_attn_res",
    "block_append", "attn_norm", "attn_fan", "kda_wq", "kda_wk",
    "kda_wv", "kda_f_a", "kda_f_b", "kda_beta", "output_gate",
    "sa", "sa_fan", "attn_out_norm", "attn_gate", "state_store", "wo",
    "attn_add", "ffn_bridge", "mlp_res", "ffn_norm", "ffn_fan",
    "dense_up", "dense_down", "gate", "routed_down", "dispatch",
    "combine", "routed_norm", "routed_up", "sw_up", "sw_down",
    "moe_add", "ffn_add", "kv_cache_fan", "conv_state_fan",
)


def _place_comm_tensor_memories(g, placement):
    """Explicitly place every communication tensor in the current graph."""
    comms = [
        kernel for kernel in g.topological_sort()
        if isinstance(kernel, CommKernel)
    ]
    comm_set = set(comms)
    propagation = defaultdict(list)

    def connect(left, right):
        propagation[left].append(right)
        propagation[right].append(left)

    for comm in comms:
        for edge in g._in_edges(comm):
            for output_name, input_name in edge.mapping.items():
                source = edge.src.outputs[output_name]
                target = comm.inputs[input_name]
                if edge.src in comm_set:
                    connect(source, target)
                else:
                    memory = placement.get_tensor_memory(source)
                    if memory is not None \
                            and placement.get_tensor_memory(target) is None:
                        placement.set_tensor_memory(target, memory)
        for edge in g._out_edges(comm):
            for output_name, input_name in edge.mapping.items():
                source = comm.outputs[output_name]
                target = edge.dst.inputs[input_name]
                if edge.dst in comm_set:
                    continue
                memory = placement.get_tensor_memory(target)
                if memory is not None \
                        and placement.get_tensor_memory(source) is None:
                    placement.set_tensor_memory(source, memory)

        if isinstance(comm, (Gather, Reduce)):
            anchor = next(iter(comm.inputs.values()), None)
            targets = comm.outputs.values()
        elif isinstance(comm, (Scatter, Broadcast)):
            anchor = next(iter(comm.outputs.values()), None)
            targets = comm.inputs.values()
        else:
            continue
        for tensor in targets:
            propagation[anchor].append(tensor)

    queue = deque(
        tensor
        for comm in comms
        for tensor in (*comm.inputs.values(), *comm.outputs.values())
        if placement.get_tensor_memory(tensor) is not None
    )
    while queue:
        source = queue.popleft()
        memory = placement.get_tensor_memory(source)
        for target in propagation[source]:
            if placement.get_tensor_memory(target) is None:
                placement.set_tensor_memory(target, memory)
                queue.append(target)


def _cluster_resources(hw):
    """Return ordered GPUs and per-node CPU/DRAM/SSD resources."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and c.kind == "gpu"],
        key=lambda c: (
            int(c.name.split("-")[0][1:]), int(c.name.rsplit("-", 1)[1])))
    cpus = defaultdict(list)
    drams = defaultdict(list)
    ssds = defaultdict(list)
    for component in hw.nodes:
        prefix = component.name.split("-", 1)[0]
        if isinstance(component, Compute) \
                and component.kind == "cpu":
            cpus[prefix].append(component)
        elif isinstance(component, Memory) and component.kind == "dram":
            drams[prefix].append(component)
        elif isinstance(component, Memory) and component.kind == "ssd":
            ssds[prefix].append(component)
    for resources in (*cpus.values(), *drams.values(), *ssds.values()):
        resources.sort(key=lambda component: int(
            component.name.rsplit("-", 1)[1]))
    return gpus, dict(cpus), dict(drams), dict(ssds)


def _nearby_memory(device, gpus_by_node, memories_by_node):
    """Map a device to its proportional local memory within one node."""
    node = device.name.split("-", 1)[0]
    devices = gpus_by_node[node]
    memories = memories_by_node[node]
    device_rank = devices.index(device)
    return memories[device_rank * len(memories) // len(devices)]


def _record_kv_cache_footprints(
    g, placement, barrier_copies, kv_read_copies=None,
):
    """Record one batch's per-memory KV usage as trace metadata."""
    for barrier in barrier_copies:
        for input_name, tensor in barrier.inputs.items():
            if not input_name.startswith("kv"):
                continue
            placement.record_memory_footprint(
                placement.get_tensor_memory(tensor),
                tensor.size_bytes,
                "kv_cache",
            )

    pending = [
        kv_read
        for copies in kv_read_copies or ()
        for kv_read in copies
    ]
    visited = set()
    roots = set()
    while pending:
        kernel = pending.pop()
        if kernel in visited:
            continue
        visited.add(kernel)
        predecessors = [edge.src for edge in g._in_edges(kernel)]
        if predecessors:
            pending.extend(predecessors)
        else:
            roots.add(kernel)
    for root in roots:
        for tensor in root.inputs.values():
            placement.record_memory_footprint(
                placement.get_tensor_memory(tensor),
                tensor.size_bytes,
                "kv_cache",
            )


def _kv_layer_id(input_name):
    """Extract the layer id from kvN and kvN_conv barrier ports."""
    return int(input_name[2:].split("_", 1)[0])


def _cache_persistence_split(kernel, n):
    """Shard persistent MLA cache by context and KDA state by batch.

    KDA decode uses the CP degree as an additional batch shard, so its
    recurrent and convolution states must retain that layout at the
    persistence boundary.  MLA remains sequence/context sharded.
    """
    output_ports = {
        port for port in kernel.inputs
        if port in ("prefill_output", "decode_output")
    } if isinstance(kernel, Nop) else set()
    if len(output_ports) != 1:
        raise TypeError("cache persistence split requires its barrier Nop")
    output_port = next(iter(output_ports))

    def shard_shape(shape, dim):
        result = list(shape)
        result[dim] //= n
        return tuple(result)

    prev_comms = {}
    for port, tensor in kernel.inputs.items():
        if port == output_port:
            comm = Broadcast(tensor.size_bytes, n)
            comm.inputs = {"x": Tensor(tensor.dtype, tensor.shape)}
            comm.outputs = {
                f"o{rank}": Tensor(tensor.dtype, tensor.shape)
                for rank in range(n)
            }
        else:
            # KDA recurrent/conv state tensors have rank >= 4.  MLA cache is
            # [B, S, D], so rank distinguishes the two without model globals.
            dim = 0 if len(tensor.shape) >= 4 else 1
            shard = shard_shape(tensor.shape, dim)
            comm = Scatter(tensor.size_bytes, n, dim=dim)
            comm.inputs = {"x": Tensor(tensor.dtype, tensor.shape)}
            comm.outputs = {
                f"o{rank}": Tensor(tensor.dtype, shard)
                for rank in range(n)
            }
        prev_comms[port] = comm

    copies = []
    for _ in range(n):
        copy = Nop()
        copy.inputs = {
            port: Tensor(
                tensor.dtype,
                tensor.shape if port == output_port else shard_shape(
                    tensor.shape, 0 if len(tensor.shape) >= 4 else 1),
            )
            for port, tensor in kernel.inputs.items()
        }
        copy.outputs = {
            port: Tensor(tensor.dtype, tensor.shape)
            for port, tensor in kernel.outputs.items()
        }
        copies.append(copy)

    next_comms = {}
    for port, tensor in kernel.outputs.items():
        comm = Gather(tensor.size_bytes, n, dim=0)
        comm.inputs = {
            f"i{rank}": Tensor(tensor.dtype, tensor.shape)
            for rank in range(n)
        }
        comm.outputs = {"y": Tensor(tensor.dtype, tensor.shape)}
        next_comms[port] = comm
    return prev_comms, copies, next_comms


def _validate_args(
    layers, batch_size, seq_prefill, is_prefill,
    cp, dp, ep, pp_partition, n_gpus,
):
    """Validate parallel degrees and stage-dependent split divisibility."""
    if sum(pp_partition) != len(layers):
        raise ValueError(
            f"pp_partition must sum to the model layer count; "
            f"got {sum(pp_partition)} for {len(layers)} layers")
    pp_degree = len(pp_partition)
    if cp * dp != ep:
        raise ValueError(
            f"CP/DP/EP placement requires cp * dp == ep; "
            f"got {cp} * {dp} != {ep}")
    if ep * pp_degree != n_gpus:
        raise ValueError(
            f"EP/PP placement requires ep * PP == n_gpus; "
            f"got {ep} * {pp_degree} != {n_gpus}")
    if N_EXPERTS % ep != 0:
        raise ValueError(
            f"N_EXPERTS={N_EXPERTS} must be divisible by ep={ep}")

    batch_degree = dp if is_prefill else cp * dp
    if batch_size % batch_degree != 0:
        raise ValueError(
            f"batch size {batch_size} must be divisible by "
            f"{'dp' if is_prefill else 'cp * dp'}={batch_degree}")

    if seq_prefill % cp != 0:
        raise ValueError(
            f"sequence/context length {seq_prefill} must be divisible by "
            f"cp={cp}")


def _set_expert_weight_read_fraction(
    layers, batch_size, context_length, ep,
):
    """Set expected expert-weight reads without changing resident weights.

    Routed-token dimensions are balanced fractionally across all experts.
    Each EP rank reads at most one full copy of every local expert weight, and
    no more expert weights than the number of routes it receives.
    """
    local_experts = N_EXPERTS // ep
    routes_per_ep_rank = Fraction(
        batch_size * context_length * TOPK, ep)
    activated_experts = min(local_experts, routes_per_ep_rank)
    read_fraction = Fraction(activated_experts, local_experts)
    for layer in layers:
        layer._expert_weight_read_fraction = read_fraction
        for kernel in layer.experts:
            kernel.weight_read_fraction = read_fraction
    return read_fraction


def _split_prefill_state(g, layer_copies, split, degree, attr_suffix):
    """Apply one one-dimensional split to every current non-expert copy."""
    result = []
    for layer, current in layer_copies:
        next_fields = {}
        for name, kernels in current.items():
            copies = []
            for kernel in kernels:
                _, split_copies, _ = g.split_kernel(
                    split, kernel, degree)
                copies.extend(split_copies)
            next_fields[name] = copies
            setattr(layer, f"_{name}_{attr_suffix}_copies", copies)
        result.append((layer, next_fields))
    return result


def _context_split_kda(g, layer_copies, dp, cp):
    """Complete KDA context splitting with summaries and prefix merges.

    Each CP rank keeps the ordinary local chunkwise KDA kernel.  The added
    kernels construct a composable local transition, exchange transitions
    within each DP group, and materialize the state/conv halo preceding that
    rank's sequence shard.
    """
    if cp == 1:
        return

    summary_elements = KDA_HD * (KDA_HD + KDA_HD)
    halo_shape = (H, 3, KDA_HD, KDA_CONV - 1)
    for layer, fields in layer_copies:
        if not layer.is_kda:
            continue
        sa_copies = fields["sa"]
        summaries = []
        merges = []
        allgathers = []
        for dp_rank in range(dp):
            group = sa_copies[dp_rank * cp:(dp_rank + 1) * cp]
            local_summaries = []
            for rank, sa in enumerate(group):
                summary = KimiK3DeltaAttnCpSummary(
                    sa.B, H, sa.S, KDA_HD, KDA_HD,
                    rank, cp, KDA_CHUNK, KDA_CONV,
                    "bf16", "bf16")
                summary.outputs = {
                    "summary": Tensor(
                        "fp32", (sa.B, H, summary_elements)),
                }
                if rank < cp - 1:
                    summary.outputs["conv_halo"] = Tensor(
                        "bf16", (sa.B, *halo_shape))
                g.add_kernel(summary)
                for edge in list(g._in_edges(sa)):
                    g.add_control_edge(edge.src, summary)
                local_summaries.append(summary)
                summaries.append(summary)

            sample = local_summaries[0].outputs["summary"]
            allgather = AllGather(sample.size_bytes * cp, cp)
            allgather.inputs = {
                f"i{rank}": Tensor(sample.dtype, sample.shape)
                for rank in range(cp)
            }
            allgather.outputs = {
                f"o{rank}": Tensor(
                    sample.dtype, (sample.shape[0], cp, *sample.shape[1:]))
                for rank in range(cp)
            }
            g.add_kernel(allgather)
            allgathers.append(allgather)
            for rank, summary in enumerate(local_summaries):
                g.add_data_edge(
                    summary, allgather, {"summary": f"i{rank}"})

            for rank, sa in enumerate(group):
                merge = KimiK3DeltaAttnCpMerge(
                    sa.B, H, KDA_HD, KDA_HD, rank, cp,
                    "bf16", "fp32")
                merge.inputs = {
                    "summaries": Tensor(
                        sample.dtype,
                        (sample.shape[0], cp, *sample.shape[1:])),
                }
                merge.outputs = {
                    "initial_state": Tensor(
                        "bf16",
                        (sa.B, H, KDA_HD, KDA_HD)),
                }
                g.add_kernel(merge)
                g.add_data_edge(
                    allgather, merge, {f"o{rank}": "summaries"})
                sa.inputs["initial_state"] = Tensor(
                    "bf16", (sa.B, H, KDA_HD, KDA_HD))
                sa.cp_rank = rank
                g.add_data_edge(
                    merge, sa, {"initial_state": "initial_state"})
                if rank > 0:
                    sa.inputs["conv_halo"] = Tensor(
                        "bf16", (sa.B, *halo_shape))
                    g.add_data_edge(
                        local_summaries[rank - 1], sa,
                        {"conv_halo": "conv_halo"})
                merges.append(merge)

        # Place halo producers before their remote consumers. Placement then
        # propagates the producer's HBM to the consumer input, and the runtime
        # resolves the direct rank-(r-1) -> rank-r remote-read path.
        original_fields = tuple(fields.items())
        fields.clear()
        fields["kda_cp_summary"] = summaries
        fields["kda_cp_merge"] = merges
        fields.update(original_fields)
        layer._kda_cp_allgathers = allgathers
        layer._kda_cp_summary_cp_dp_copies = summaries
        layer._kda_cp_merge_cp_dp_copies = merges


def _place_experts_and_routes(
    g, layer, fields, devices, placement, hw, *, shard_experts,
):
    """Place expert weights and, for EP, bind routed buffers to owners."""

    if not layer.experts:
        return

    output_consumers = {}
    input_producers = {}

    def consumers_by_output(kernel):
        if kernel not in output_consumers:
            consumers = defaultdict(list)
            for edge in g._out_edges(kernel):
                for output_name, input_name in edge.mapping.items():
                    consumers[output_name].append(
                        edge.dst.inputs[input_name])
            output_consumers[kernel] = consumers
        return output_consumers[kernel]

    def producers_by_input(kernel):
        if kernel not in input_producers:
            producers = defaultdict(list)
            for edge in g._in_edges(kernel):
                for output_name, input_name in edge.mapping.items():
                    producers[input_name].append(
                        edge.src.outputs[output_name])
            input_producers[kernel] = producers
        return input_producers[kernel]

    def set_output_memory(kernel, output_name, memory):
        placement.set_tensor_memory(kernel.outputs[output_name], memory)
        for tensor in consumers_by_output(kernel)[output_name]:
            placement.set_tensor_memory(tensor, memory)

    def set_input_memory(kernel, input_name, memory):
        placement.set_tensor_memory(kernel.inputs[input_name], memory)
        for tensor in producers_by_input(kernel)[input_name]:
            placement.set_tensor_memory(tensor, memory)

    local_experts = N_EXPERTS // len(devices) if shard_experts else N_EXPERTS
    for expert_id in range(N_EXPERTS):
        owner = expert_id // local_experts if shard_experts else 0
        owner_device = devices[owner]
        up = layer.experts[expert_id * 2]
        down = layer.experts[expert_id * 2 + 1]
        placement.set_kernel_device(up, owner_device)
        placement.set_kernel_device(down, owner_device)
        set_input_memory(
            up, "x", hw.find_local_memory(owner_device))

    if not shard_experts:
        return
    for dispatch in fields["dispatch"]:
        for expert_id in range(N_EXPERTS):
            owner = expert_id // local_experts
            set_output_memory(
                dispatch, f"o{expert_id}",
                hw.find_local_memory(devices[owner]))
    for combine in fields["combine"]:
        for expert_id in range(N_EXPERTS):
            owner = expert_id // local_experts
            set_input_memory(
                combine, f"i{expert_id}",
                hw.find_local_memory(devices[owner]))


def _place_pp_boundary_spawn(g, placement, spawn, memory):
    """Write a PP boundary once into the destination stage's local HBM."""
    for tensor in (*spawn.inputs.values(), *spawn.outputs.values()):
        placement.set_tensor_memory(tensor, memory)
    for edge in g._in_edges(spawn):
        for output_name, input_name in edge.mapping.items():
            placement.set_tensor_memory(
                edge.src.outputs[output_name], memory)
            placement.set_tensor_memory(
                spawn.inputs[input_name], memory)
    for edge in g._out_edges(spawn):
        for output_name, input_name in edge.mapping.items():
            placement.set_tensor_memory(
                spawn.outputs[output_name], memory)
            placement.set_tensor_memory(
                edge.dst.inputs[input_name], memory)


def _insert_pp_block_residual_sends(
    g, layers, layer_stages, block_in_fans_by_layer, bridges_by_layer,
):
    """Materialize AttnRes state at PP edges once the main path is ready."""
    for layer_id in range(1, len(layers)):
        if layer_stages[layer_id] == layer_stages[layer_id - 1]:
            continue
        block_in_fans = block_in_fans_by_layer[layer_id]
        bridges = bridges_by_layer[layer_id]
        if len(block_in_fans) != len(bridges):
            raise ValueError(
                f"Layer {layer_id} has {len(block_in_fans)} block residual "
                f"ranks but {len(bridges)} main-path ranks")
        sends = []
        for spawn, bridge in zip(block_in_fans, bridges):
            edge = g._in_edges(spawn)[0]
            # A split-generated communication kernel already materializes its
            # output according to the destination consumer placement. Adding
            # a Send after it would leave the intermediate comm-to-comm tensor
            # without a memory anchor (for example Scatter -> Send -> Spawn).
            if isinstance(edge.src, CommKernel):
                materialization = edge.src
            else:
                source_port = next(
                    output_name
                    for output_name, input_name in edge.mapping.items()
                    if input_name == "x")
                tensor = spawn.inputs["x"]
                send = Send(tensor.size_bytes)
                send.inputs = {"x": Tensor(tensor.dtype, tensor.shape)}
                send.outputs = {"y": Tensor(tensor.dtype, tensor.shape)}
                g.insert_identity(
                    send, edge.src, spawn, {source_port: "x"})
                sends.append(send)
                materialization = send

            # The block residual can be produced near the beginning of the
            # source stage and bypass many layers. Delay its PP transfer until
            # the ordinary hidden-state path reaches the same boundary. This
            # keeps the destination stage from starting with an early transfer
            # followed by a long dependency bubble before AttnRes.
            main_source = g._in_edges(bridge)[0].src
            g.add_control_edge(main_source, materialization)
        layers[layer_id]._pp_block_residual_sends = sends


def optimize_model_cluster_prefill(
    g, layers, hw, emb=None, read_input=None, output_head=None, *,
    cp, dp, ep, pp_partition, n_gpus,
):
    """Apply dynamic DP→CP→EP→PP placement to a prefill-only graph.

    Non-expert kernels use CP×DP within their assigned PP stage. Experts use
    EP within the stage, with EP=CP×DP. Each pp_partition entry is the number
    of model layers assigned to that stage. PP models one prefill wave without
    a microbatch schedule.
    """
    batch_size, seq_prefill = emb.outputs["y"].shape[:2]
    _validate_args(
        layers, batch_size, seq_prefill, True,
        cp, dp, ep, pp_partition, n_gpus)
    _set_expert_weight_read_fraction(
        layers, batch_size, seq_prefill, ep)
    pp_degree = len(pp_partition)
    layer_stages = [
        stage
        for stage, layer_count in enumerate(pp_partition)
        for _ in range(layer_count)
    ]
    gpus, cpus, drams, _ = _cluster_resources(hw)

    kv_barrier = layers[0].kv_persist_barrier

    layer_copies = []
    for layer in layers:
        fields = {
            name: [getattr(layer, name)]
            for name in _PREFILL_PARALLEL_FIELDS
            if getattr(layer, name) is not None
        }
        layer_copies.append((layer, fields))
    emb_copies = [emb]

    # Finish every graph transform before assigning any placement.
    _, emb_copies, _ = g.split_kernel(batch_split, emb, dp)
    layer_copies = _split_prefill_state(
        g, layer_copies, batch_split, dp, "dp")
    output_head_dp_copies = []
    for kernel in output_head:
        _, copies, _ = g.split_kernel(batch_split, kernel, dp)
        output_head_dp_copies.append(copies)
    _, barrier_dp_copies, _ = g.split_kernel(
        batch_split, kv_barrier, dp)

    cp_emb_copies = []
    for copy in emb_copies:
        _, split_copies, _ = g.split_kernel(
            context_split_prefill, copy, cp)
        cp_emb_copies.extend(split_copies)
    emb_copies = cp_emb_copies
    layer_copies = _split_prefill_state(
        g, layer_copies, context_split_prefill, cp, "cp_dp")
    barrier_copies = []
    for copy in barrier_dp_copies:
        _, split_copies, _ = g.split_kernel(
            _cache_persistence_split, copy, cp)
        barrier_copies.extend(split_copies)
    for layer in layers:
        layer._kv_persist_barrier_cp_dp_copies = barrier_copies

    optimize_comms(g)
    # KDA summaries have no tensor inputs of their own, so their readiness is
    # expressed with control edges from the local attention predecessors.
    # Add those edges only after communication fusion has finalized those
    # predecessors; otherwise fusion can remove them and make summaries roots.
    _context_split_kda(g, layer_copies, dp, cp)

    # PP changes placement only; it does not split kernels or modify the graph.
    stage_gpus = [
        gpus[stage * ep:(stage + 1) * ep]
        for stage in range(pp_degree)
    ]
    copies_by_layer = {id(layer): fields for layer, fields in layer_copies}

    # A block residual is carried through a chain of zero-copy Spawn aliases
    # independently of the ordinary hidden-state bridge.  At a PP boundary,
    # insert one explicit point-to-point materialization so the source alias
    # remains in its stage HBM while the destination chain starts locally.
    _insert_pp_block_residual_sends(
        g, layers, layer_stages,
        [
            copies_by_layer[id(layer)].get("block_in_fan", ())
            for layer in layers
        ],
        [
            copies_by_layer[id(layer)].get("bridge", ())
            for layer in layers
        ],
    )

    # Place every compute kernel and tensor only after all splits are complete.
    placement = Placement(hardware=hw, graph=g)
    if read_input is not None:
        placement.set_kernel_device(read_input, stage_gpus[0][0])
        node = stage_gpus[0][0].name.split("-", 1)[0]
        placement.set_tensor_memory(
            read_input.inputs["tokens"],
            hw.find_local_memory(cpus[node][0]))
    for rank, copy in enumerate(emb_copies):
        placement.set_kernel_device(copy, stage_gpus[0][rank])

    for layer_id, layer in enumerate(layers):
        devices = stage_gpus[layer_stages[layer_id]]
        fields = copies_by_layer[id(layer)]
        for copies in fields.values():
            for rank, copy in enumerate(copies):
                placement.set_kernel_device(copy, devices[rank])
        _place_experts_and_routes(
            g, layer, fields, devices, placement, hw, shard_experts=True)

    # At a PP boundary, materialize the ordinary hidden-state layer input in
    # destination HBM once.  The independently carried AttnRes block residual
    # is handled by the explicit Send inserted above.
    for layer_id in range(1, len(layers)):
        stage = layer_stages[layer_id]
        if stage == layer_stages[layer_id - 1]:
            continue
        bridges = copies_by_layer[id(layers[layer_id])]["bridge"]
        for rank, bridge in enumerate(bridges):
            _place_pp_boundary_spawn(
                g, placement, bridge,
                hw.find_local_memory(stage_gpus[stage][rank]))

    final_devices = stage_gpus[layer_stages[-1]]
    for copies in output_head_dp_copies:
        for dp_rank, copy in enumerate(copies):
            placement.set_kernel_device(
                copy, final_devices[dp_rank * cp + cp - 1])

    for rank, barrier in enumerate(barrier_copies):
        placement.set_kernel_device(barrier, final_devices[rank])
        for input_name, tensor in barrier.inputs.items():
            if input_name.startswith("kv"):
                layer_id = _kv_layer_id(input_name)
                devices = stage_gpus[layer_stages[layer_id]]
            else:
                devices = final_devices
            placement.set_tensor_memory(
                tensor, hw.find_local_memory(devices[rank]))
        output_memory = hw.find_local_memory(final_devices[rank])
        for tensor in barrier.outputs.values():
            placement.set_tensor_memory(tensor, output_memory)

    _place_comm_tensor_memories(g, placement)

    g.validate()
    placement.validate(g)
    _record_kv_cache_footprints(g, placement, barrier_copies)
    return g, placement


def optimize_model_cluster_decode(
    g, layers, hw, emb=None, read_input=None, kv_cache_reads=None,
    output_head=None, *, seq_prefill, cp, dp, ep, pp_partition, n_gpus,
):
    """Apply CP, DP, EP, and PP to a one-step decode-only graph.

    KV is a persistent read-only model input. The optimizer does not construct
    cache slices, append the current token, or model cache ownership changes.
    """
    output_attn_res, final_norm, logits, sampling = output_head
    batch_size = emb.outputs["y"].shape[0]
    _validate_args(
        layers, batch_size, seq_prefill, False,
        cp, dp, ep, pp_partition, n_gpus)
    _set_expert_weight_read_fraction(layers, batch_size, 1, ep)
    pp_degree = len(pp_partition)

    layer_stages = [
        stage
        for stage, layer_count in enumerate(pp_partition)
        for _ in range(layer_count)
    ]
    gpus, cpus, drams, ssds = _cluster_resources(hw)
    gpus_by_node = defaultdict(list)
    for device in gpus:
        gpus_by_node[device.name.split("-", 1)[0]].append(device)
    stage_gpus = [
        gpus[stage * ep:(stage + 1) * ep]
        for stage in range(pp_degree)
    ]

    # DP comes first so every later CP split operates within one independent
    # batch shard. Communication kernels are created only as boundaries of
    # compute splits; existing communication kernels are never split directly.
    _, emb_dp_copies, _ = g.split_kernel(batch_split, emb, dp)
    _, output_attn_res_dp_copies, _ = g.split_kernel(
        batch_split, output_attn_res, dp)
    _, final_norm_dp_copies, _ = g.split_kernel(
        batch_split, final_norm, dp)
    _, logits_dp_copies, _ = g.split_kernel(batch_split, logits, dp)
    _, sampling_dp_copies, _ = g.split_kernel(batch_split, sampling, dp)

    dp_fields_by_layer = []
    copies_by_layer = []
    for layer in layers:
        dp_fields = {}
        if layer.is_kda:
            field_names = _DECODE_KDA_BATCH_FIELDS
            dp_only_names = ()
        else:
            field_names = (
                *_DECODE_MLA_DP_ONLY_FIELDS,
                *_DECODE_MLA_REPLICATED_FIELDS,
                *_DECODE_MLA_SHARDED_FIELDS,
                "sa", "kv_cache_fan",
            )
            dp_only_names = _DECODE_MLA_DP_ONLY_FIELDS
        for name in field_names:
            kernel = getattr(layer, name)
            if kernel is None or name in dp_fields:
                continue
            _, copies, _ = g.split_kernel(
                batch_split, kernel, dp)
            dp_fields[name] = copies
        layer_copies = {
            "dp": {
                name: dp_fields[name]
                for name in dp_only_names if name in dp_fields
            },
            "cp_dp": {},
        }
        layer._decode_copies = layer_copies
        dp_fields_by_layer.append(dp_fields)
        copies_by_layer.append(layer_copies)

    kv_read_dp_groups = []
    for layer_id, (layer, kv_read) in enumerate(
            zip(layers, kv_cache_reads)):
        reads = [kv_read]
        if layer.conv_state_read is not None:
            reads.append(layer.conv_state_read)
        for cache_read in reads:
            _, copies, _ = g.split_kernel(batch_split, cache_read, dp)
            kv_read_dp_groups.append((layer_id, layer.is_kda, copies))

    barrier = layers[0].kv_persist_barrier
    _, barrier_dp_copies, _ = g.split_kernel(batch_split, barrier, dp)

    # Restore each independent DP subgraph before CP is introduced.
    optimize_comms(g)

    # Apply CP separately inside every DP subgraph. Each attention split now
    # creates its own Q Broadcast and output Reduce for that DP group.
    kv_read_cp_dp_groups = []
    for layer_id, is_kda, dp_copies in kv_read_dp_groups:
        cp_dp_copies = []
        for copy in dp_copies:
            split = batch_split if is_kda else context_split_decode
            _, copies, _ = g.split_kernel(split, copy, cp)
            cp_dp_copies.extend(copies)
        kv_read_cp_dp_groups.append((layer_id, cp_dp_copies))

    barrier_copies = []
    for copy in barrier_dp_copies:
        _, copies, _ = g.split_kernel(_cache_persistence_split, copy, cp)
        barrier_copies.extend(copies)
    for layer in layers:
        layer._kv_persist_barrier_cp_dp_copies = barrier_copies

    for layer, dp_fields, layer_copies in zip(
            layers, dp_fields_by_layer, copies_by_layer):
        if layer.is_kda:
            cp_fields = {name: [] for name in dp_fields}
            for dp_rank in range(dp):
                for name, dp_copies in dp_fields.items():
                    _, copies, _ = g.split_kernel(
                        batch_split, dp_copies[dp_rank], cp)
                    cp_fields[name].extend(copies)
            layer_copies["cp_dp"] = cp_fields
            continue

        cp_field_names = (
            *_DECODE_MLA_REPLICATED_FIELDS,
            *_DECODE_MLA_SHARDED_FIELDS,
            "sa", "kv_cache_fan",
        )
        cp_fields = {
            name: [] for name in cp_field_names if name in dp_fields
        }
        for dp_rank in range(dp):
            _, copies, _ = g.split_kernel(
                context_split_decode,
                dp_fields["kv_cache_fan"][dp_rank], cp)
            cp_fields["kv_cache_fan"].extend(copies)

            sa_prev, copies, _ = g.split_kernel(
                context_split_decode, dp_fields["sa"][dp_rank], cp)
            cp_fields["sa"].extend(copies)

            q_broadcast = sa_prev["q"]
            for name in reversed(_DECODE_MLA_REPLICATED_FIELDS):
                q_broadcast, copies = g.dup(
                    general_dup, dp_fields[name][dp_rank])
                cp_fields[name].extend(copies)

            for name in _DECODE_MLA_SHARDED_FIELDS:
                if name not in dp_fields:
                    continue
                _, copies, _ = g.split_kernel(
                    batch_split, dp_fields[name][dp_rank], cp)
                cp_fields[name].extend(copies)

        layer_copies["cp_dp"] = cp_fields

    optimize_comms(g)

    # Decode carries the AttnRes block state outside the ordinary hidden-state
    # bridge just like prefill. MLA keeps this fan-out DP-only while KDA splits
    # it across CP x DP, so collect both layouts before inserting PP sends.
    _insert_pp_block_residual_sends(
        g, layers, layer_stages,
        [
            (
                *layer_copies["dp"].get("block_in_fan", ()),
                *layer_copies["cp_dp"].get("block_in_fan", ()),
            )
            for layer_copies in copies_by_layer
        ],
        [
            (
                *layer_copies["dp"].get("bridge", ()),
                *layer_copies["cp_dp"].get("bridge", ()),
            )
            for layer_copies in copies_by_layer
        ],
    )

    # Treat KV loading as a preload phase. The token reader marks the start of
    # the measured decode step and cannot run until every DP×CP KV shard has
    # been materialized in its destination HBM.
    for _, copies in kv_read_cp_dp_groups:
        for kv_read in copies:
            g.add_control_edge(kv_read, read_input)

    placement = Placement(hardware=hw, graph=g)

    # Keep the persistent KV source on SSD and materialize each CP×DP shard in
    # its destination HBM during the preload phase.  The later comm-memory
    # propagation carries this SSD placement through split-generated root
    # Scatter kernels, so the unsplit external inputs do not consume DRAM.
    for layer_id, copies in kv_read_cp_dp_groups:
        devices = stage_gpus[layer_stages[layer_id]]
        for rank, kernel in enumerate(copies):
            device = devices[rank]
            placement.set_kernel_device(kernel, device)
            source_tensor = next(iter(kernel.inputs.values()))
            placement.set_tensor_memory(
                source_tensor,
                _nearby_memory(device, gpus_by_node, ssds))

    first_devices = stage_gpus[0]
    placement.set_kernel_device(read_input, first_devices[0])
    placement.set_tensor_memory(
        read_input.inputs["tokens"],
        _nearby_memory(first_devices[0], gpus_by_node, drams))
    for dp_rank, kernel in enumerate(emb_dp_copies):
        placement.set_kernel_device(
            kernel, first_devices[dp_rank * cp])

    final_devices = stage_gpus[layer_stages[-1]]
    for copies in (
        output_attn_res_dp_copies,
        final_norm_dp_copies,
        logits_dp_copies,
        sampling_dp_copies,
    ):
        for dp_rank, kernel in enumerate(copies):
            placement.set_kernel_device(
                kernel, final_devices[dp_rank * cp])

    for layer_id, (layer, layer_copies) in enumerate(
            zip(layers, copies_by_layer)):
        devices = stage_gpus[layer_stages[layer_id]]
        for copies in layer_copies["dp"].values():
            for dp_rank, kernel in enumerate(copies):
                placement.set_kernel_device(
                    kernel, devices[dp_rank * cp])

        for copies in layer_copies["cp_dp"].values():
            for rank, kernel in enumerate(copies):
                placement.set_kernel_device(kernel, devices[rank])

        if layer.experts:
            route_fields = {
                name: layer_copies["cp_dp"][name]
                for name in ("dispatch", "combine")
            }
            _place_experts_and_routes(
                g, layer, route_fields, devices, placement, hw,
                shard_experts=True)

    for layer_id in range(1, len(layers)):
        stage = layer_stages[layer_id]
        if stage == layer_stages[layer_id - 1]:
            continue
        layer_copies = copies_by_layer[layer_id]
        if "bridge" in layer_copies["dp"]:
            bridges = layer_copies["dp"]["bridge"]
            bridge_devices = stage_gpus[stage][::cp]
        else:
            bridges = layer_copies["cp_dp"]["bridge"]
            bridge_devices = stage_gpus[stage]
        for bridge, device in zip(bridges, bridge_devices):
            _place_pp_boundary_spawn(
                g, placement, bridge, hw.find_local_memory(device))

    for rank, barrier_copy in enumerate(barrier_copies):
        placement.set_kernel_device(barrier_copy, final_devices[rank])
        for input_name, tensor in barrier_copy.inputs.items():
            if input_name.startswith("kv"):
                layer_id = _kv_layer_id(input_name)
                devices = stage_gpus[layer_stages[layer_id]]
            else:
                devices = final_devices
            placement.set_tensor_memory(
                tensor, hw.find_local_memory(devices[rank]))
        output_memory = hw.find_local_memory(final_devices[rank])
        for tensor in barrier_copy.outputs.values():
            placement.set_tensor_memory(tensor, output_memory)

    _place_comm_tensor_memories(g, placement)
    g.validate()
    placement.validate(g)
    _record_kv_cache_footprints(
        g, placement, barrier_copies,
        [copies for _, copies in kv_read_cp_dp_groups])
    return g, placement


def optimize_model_superchip(g, hw):
    """Place all kernels on the single fused GPU (no splits, no comms)."""
    gpu = [c for c in hw.nodes if isinstance(c, Compute)
           and c.kind == "gpu"][0]
    p = Placement(hardware=hw, graph=g)
    for k in g.topological_sort():
        p.set_kernel_device(k, gpu)
    g.validate()
    p.validate(g)
    return g, p
