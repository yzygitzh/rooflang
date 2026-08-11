"""DeepSeek V4 Pro inference — graph splitting and placement strategies."""

from collections import defaultdict

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import (
    Broadcast, CommKernel, Gather, Reduce, Scatter,
)
from rooflang.language.kernels.identity import Move
from rooflang.language.optimization.comm import (
    canonicalize_split_comms, optimize_comms,
)
from rooflang.language.optimization.split import (
    batch_split, batch_split_comm, context_split,
    decode_attention_context_split, kv_persistence_split,
    replicate_before,
)
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor

from rooflang.programs.dsv4_pro.config import (
    COMPRESS_RATIOS, N_EXPERTS, TOPK, WINDOW,
)


_PREFILL_PARALLEL_FIELDS = (
    "bridge", "attn_norm", "attn_fan", "comp", "comp_norm",
    "wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "kv_concat",
    "kv_persist_fan", "sa", "kv_win_slice", "wo_a", "wo_b",
    "attn_add", "ffn_bridge", "ffn_norm", "ffn_fan", "gate",
    "dispatch", "combine", "sw_up", "sw_down", "moe_add", "ffn_add",
)

_DECODE_REPLICATED_FIELDS = (
    "wq_a", "q_norm", "wq_b",
)

_DECODE_SHARDED_FIELDS = (
    "wo_a", "wo_b", "attn_add", "ffn_bridge", "ffn_norm", "ffn_fan",
    "gate", "dispatch", "combine", "sw_up", "sw_down", "moe_add",
    "ffn_add",
)


def _tag_split_comms(split, axis):
    """Tag only the wrapper comms introduced by one logical split axis."""
    def tagged_split(kernel, n):
        prev_comms, copies, next_comms = split(kernel, n)
        for comm in (*prev_comms.values(), *next_comms.values()):
            comm._split_axis = axis
        return prev_comms, copies, next_comms

    return tagged_split


def _place_comm_tensor_memories(g, placement):
    """Explicitly place every communication tensor in the current graph."""
    comms = [
        kernel for kernel in g.topological_sort()
        if isinstance(kernel, CommKernel)
    ]

    def place_from_edges(comm):
        for edge in g._in_edges(comm):
            for output_name, input_name in edge.mapping.items():
                memory = placement.get_tensor_memory(
                    edge.src.outputs[output_name])
                tensor = comm.inputs[input_name]
                if memory is not None \
                        and placement.get_tensor_memory(tensor) is None:
                    placement.set_tensor_memory(tensor, memory)
        for edge in g._out_edges(comm):
            for output_name, input_name in edge.mapping.items():
                memory = placement.get_tensor_memory(
                    edge.dst.inputs[input_name])
                tensor = comm.outputs[output_name]
                if memory is not None \
                        and placement.get_tensor_memory(tensor) is None:
                    placement.set_tensor_memory(tensor, memory)

    def place_root(comm):
        if isinstance(comm, (Gather, Reduce)):
            anchor = next(iter(comm.inputs.values()), None)
            targets = comm.outputs.values()
        elif isinstance(comm, (Scatter, Broadcast)):
            anchor = next(iter(comm.outputs.values()), None)
            targets = comm.inputs.values()
        else:
            return
        memory = placement.get_tensor_memory(anchor)
        if memory is None:
            return
        for tensor in targets:
            if placement.get_tensor_memory(tensor) is None:
                placement.set_tensor_memory(tensor, memory)

    for comm in comms:
        place_from_edges(comm)
        place_root(comm)
    for comm in reversed(comms):
        place_from_edges(comm)
        place_root(comm)


def _cluster_a_resources(hw):
    """Return ordered GPUs and per-node CPU/DRAM resources."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and "nvidia-b300" in c.name],
        key=lambda c: (
            int(c.name.split("-")[0][1:]), int(c.name.rsplit("-", 1)[1])))
    cpus = defaultdict(list)
    drams = defaultdict(list)
    for component in hw.nodes:
        prefix = component.name.split("-", 1)[0]
        if isinstance(component, Compute) \
                and "intel-xeon" in component.name:
            cpus[prefix].append(component)
        elif isinstance(component, Memory) and "-ddr5-" in component.name:
            drams[prefix].append(component)
    for resources in (*cpus.values(), *drams.values()):
        resources.sort(key=lambda component: int(
            component.name.rsplit("-", 1)[1]))
    return gpus, dict(cpus), dict(drams)


def _validate_cp_dp_ep_pp_prefill(
    layers, emb, cp, dp, ep, pp_partition, n_gpus,
):
    """Validate parallel degrees, PP layer counts, and split divisibility."""
    if min(cp, dp, ep, n_gpus) <= 0:
        raise ValueError("cp, dp, ep, and n_gpus must all be positive")
    if not pp_partition or any(
            not isinstance(count, int) or count <= 0
            for count in pp_partition):
        raise ValueError("pp_partition must contain positive layer counts")
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
    if emb is None:
        raise ValueError("prefill optimizer requires an embedding kernel")

    batch_size, seq_prefill = emb.outputs["y"].shape[:2]
    if batch_size % dp != 0:
        raise ValueError(
            f"batch size {batch_size} must be divisible by dp={dp}")
    if seq_prefill % cp != 0:
        raise ValueError(
            f"prefill sequence {seq_prefill} must be divisible by cp={cp}")
    if WINDOW % cp != 0:
        raise ValueError(f"WINDOW={WINDOW} must be divisible by cp={cp}")
    for ratio in set(COMPRESS_RATIOS):
        if seq_prefill % ratio != 0:
            raise ValueError(
                f"prefill sequence {seq_prefill} must be divisible by "
                f"compression ratio {ratio}")
        if (seq_prefill // ratio) % cp != 0:
            raise ValueError(
                f"compressed sequence {seq_prefill // ratio} must be "
                f"divisible by cp={cp}")
    routed_tokens = batch_size * seq_prefill * TOPK
    if routed_tokens % (N_EXPERTS * ep) != 0:
        raise ValueError(
            "expert-token count must be divisible by cp * dp; got "
            f"B*S*TOPK={routed_tokens}, N_EXPERTS*ep={N_EXPERTS * ep}")


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


def _place_experts_and_routes(
    g, layer, fields, devices, placement, hw, *, shard_experts,
):
    """Place expert weights and, for EP, bind routed buffers to owners."""

    def set_output_memory(kernel, output_name, memory):
        placement.set_tensor_memory(kernel.outputs[output_name], memory)
        for edge in g._out_edges(kernel):
            if output_name not in edge.mapping:
                continue
            input_name = edge.mapping[output_name]
            placement.set_tensor_memory(edge.dst.inputs[input_name], memory)

    def set_input_memory(kernel, input_name, memory):
        placement.set_tensor_memory(kernel.inputs[input_name], memory)
        for edge in g._in_edges(kernel):
            for output_name, mapped_input in edge.mapping.items():
                if mapped_input != input_name:
                    continue
                placement.set_tensor_memory(
                    edge.src.outputs[output_name], memory)

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
    _validate_cp_dp_ep_pp_prefill(
        layers, emb, cp, dp, ep, pp_partition, n_gpus)
    if output_head is None or len(output_head) != 4:
        raise ValueError(
            "prefill optimizer requires last-token output head kernels")
    pp_degree = len(pp_partition)
    layer_stages = [
        stage
        for stage, layer_count in enumerate(pp_partition)
        for _ in range(layer_count)
    ]
    gpus, cpus, drams = _cluster_a_resources(hw)
    if len(gpus) != n_gpus:
        raise ValueError(
            f"hardware contains {len(gpus)} B300 GPUs, expected {n_gpus}")
    if not cpus or not drams:
        raise ValueError("B300 Cluster A requires per-node CPUs and DRAMs")

    kv_barrier = layers[0].kv_persist_barrier
    if kv_barrier is None \
            or any(layer.kv_persist_barrier is not kv_barrier
                   for layer in layers):
        raise ValueError(
            "prefill KV persistence is missing from the declared model")

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
        _, split_copies, _ = g.split_kernel(context_split, copy, cp)
        cp_emb_copies.extend(split_copies)
    emb_copies = cp_emb_copies
    layer_copies = _split_prefill_state(
        g, layer_copies, context_split, cp, "cp_dp")
    barrier_copies = []
    for copy in barrier_dp_copies:
        _, split_copies, _ = g.split_kernel(
            kv_persistence_split, copy, cp)
        barrier_copies.extend(split_copies)
    for layer in layers:
        layer._kv_persist_barrier_cp_dp_copies = barrier_copies

    # PP changes placement only; it does not split kernels or modify the graph.
    stage_gpus = [
        gpus[stage * ep:(stage + 1) * ep]
        for stage in range(pp_degree)
    ]
    copies_by_layer = {id(layer): fields for layer, fields in layer_copies}

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

    final_devices = stage_gpus[layer_stages[-1]]
    for copies in output_head_dp_copies:
        for dp_rank, copy in enumerate(copies):
            placement.set_kernel_device(
                copy, final_devices[dp_rank * cp + cp - 1])

    for rank, barrier in enumerate(barrier_copies):
        for input_name, tensor in barrier.inputs.items():
            if input_name.startswith("kv"):
                layer_id = int(input_name[2:])
                devices = stage_gpus[layer_stages[layer_id]]
            else:
                devices = final_devices
            placement.set_tensor_memory(
                tensor, hw.find_local_memory(devices[rank]))
        output_memory = hw.find_local_memory(final_devices[rank])
        for tensor in barrier.outputs.values():
            placement.set_tensor_memory(tensor, output_memory)

    _place_comm_tensor_memories(g, placement)

    # Communication rewriting runs only after split and placement are final.
    optimize_comms(g, placement)
    g.validate()
    placement.validate(g)
    return g, placement


def optimize_model_cluster_decode(
    g, layers, hw, emb=None, read_input=None, kv_cache_reads=None,
    output_head=None, *, cp, dp, ep, pp_partition, n_gpus,
):
    """Apply CP, DP, EP, and PP to a one-step decode-only graph.

    KV is a persistent read-only model input. The optimizer does not construct
    cache slices, append the current token, or model cache ownership changes.
    """
    if emb is None or read_input is None or not kv_cache_reads \
            or output_head is None or len(output_head) != 3:
        raise ValueError("decode optimizer requires input, KV, and output head")
    if min(cp, dp, ep, n_gpus) <= 0:
        raise ValueError("cp, dp, ep, and n_gpus must all be positive")
    if not pp_partition or any(
            not isinstance(count, int) or count <= 0
            for count in pp_partition):
        raise ValueError("pp_partition must contain positive layer counts")

    final_norm, logits, sampling = output_head
    n_layers = len(layers)
    if sum(pp_partition) != n_layers:
        raise ValueError(
            "pp_partition must sum to the model layer count; "
            f"got {sum(pp_partition)} for {n_layers} layers")
    pp_degree = len(pp_partition)
    if cp * dp != ep:
        raise ValueError(
            "CP/DP/EP placement requires cp * dp == ep; "
            f"got {cp} * {dp} != {ep}")
    if ep * pp_degree != n_gpus:
        raise ValueError(
            "EP/PP placement requires ep * PP == n_gpus; "
            f"got {ep} * {pp_degree} != {n_gpus}")
    if N_EXPERTS % ep != 0:
        raise ValueError(
            f"N_EXPERTS={N_EXPERTS} must be divisible by ep={ep}")
    if len(kv_cache_reads) != n_layers:
        raise ValueError(
            f"expected {n_layers} KV cache reads, got {len(kv_cache_reads)}")

    batch_size = emb.outputs["y"].shape[0]
    if batch_size % ep != 0:
        raise ValueError(
            f"decode batch size {batch_size} must be divisible by ep={ep}")
    routed_tokens = batch_size * TOPK
    if routed_tokens % (N_EXPERTS * ep) != 0:
        raise ValueError(
            "expert-token count must be divisible by EP copies; got "
            f"B*TOPK={routed_tokens}, N_EXPERTS*ep={N_EXPERTS * ep}")

    layer_stages = [
        stage
        for stage, layer_count in enumerate(pp_partition)
        for _ in range(layer_count)
    ]
    gpus, cpus, drams = _cluster_a_resources(hw)
    if len(gpus) != n_gpus:
        raise ValueError(
            f"hardware contains {len(gpus)} B300 GPUs, expected {n_gpus}")
    if not cpus or not drams:
        raise ValueError("B300 Cluster A requires per-node CPUs and DRAMs")
    stage_gpus = [
        gpus[stage * ep:(stage + 1) * ep]
        for stage in range(pp_degree)
    ]

    for layer_id, (kv_read, layer) in enumerate(
            zip(kv_cache_reads, layers)):
        cache_len = kv_read.outputs["y"].shape[1]
        if cache_len % cp != 0:
            raise ValueError(
                f"layer {layer_id} KV length {cache_len} must be "
                f"divisible by cp={cp}")
        if layer.sa.S_kv != cache_len:
            raise ValueError(
                f"layer {layer_id} attention sees {layer.sa.S_kv} KV "
                f"entries, expected persistent cache length {cache_len}")

    # CP comes first. Attention broadcasts Q, shards the persistent KV input,
    # and reduces the partial outputs. Move the Q broadcast before the Q
    # projection chain while it is still adjacent to those kernels.
    kv_read_cp_copies = []
    for kv_read in kv_cache_reads:
        _, copies, _ = g.split_kernel(context_split, kv_read, cp)
        kv_read_cp_copies.append(copies)

    barrier = layers[0].kv_persist_barrier
    if barrier is None or any(
            layer.kv_persist_barrier is not barrier
            for layer in layers):
        raise ValueError("decode KV persistence barrier is missing")
    barrier_prev, barrier_cp_copies, _ = g.split_kernel(
        kv_persistence_split, barrier, cp)
    cp_collectives_to_replicate = [barrier_prev["decode_output"]]

    cp_fields_by_layer = []
    for layer in layers:
        cp_fields = {}
        _, cp_fields["kv_cache_fan"], _ = g.split_kernel(
            context_split, layer.kv_cache_fan, cp)
        sa_prev, cp_fields["sa"], sa_next = g.split_kernel(
            decode_attention_context_split, layer.sa, cp)

        q_broadcast = sa_prev["q"]
        for name in reversed(_DECODE_REPLICATED_FIELDS):
            q_broadcast, copies = g.dup(
                replicate_before, getattr(layer, name))
            cp_fields[name] = copies

        for name in _DECODE_SHARDED_FIELDS:
            prev_comms, copies, next_comms = g.split_kernel(
                batch_split, getattr(layer, name), cp)
            cp_fields[name] = copies
            if name == "wo_a":
                attention_scatter = prev_comms["x"]
            elif name == "attn_add":
                residual_scatter = prev_comms["a"]
            elif name == "ffn_add":
                layer_output_gather = next_comms["y"]
        cp_collectives_to_replicate.extend(
            (q_broadcast, sa_next["y"], attention_scatter,
             residual_scatter, layer_output_gather))
        cp_fields_by_layer.append(cp_fields)

    # DP is the second graph transform. Split each CP rank by batch, then
    # flatten copies in DP-major order to match the stage GPU rank layout.
    # First replicate every batch-shaped CP collective per DP group. The
    # Scatter/Gather wrappers introduced here cancel against those produced
    # when the adjacent compute copies are batch-split below.
    dp_split = _tag_split_comms(batch_split, "dp")
    dp_comm_split = _tag_split_comms(batch_split_comm, "dp")
    for kernel in g.kernels:
        if isinstance(kernel, CommKernel):
            kernel._split_axis = "cp"
    for comm in cp_collectives_to_replicate:
        g.split_kernel(dp_comm_split, comm, dp)

    def split_cp_copies(copies):
        copies_by_cp = []
        for copy in copies:
            _, dp_copies, _ = g.split_kernel(dp_split, copy, dp)
            copies_by_cp.append(dp_copies)
        return [
            copies_by_cp[cp_rank][dp_rank]
            for dp_rank in range(dp)
            for cp_rank in range(cp)
        ]

    kv_read_cp_dp_copies = []
    for layer_id, copies in enumerate(kv_read_cp_copies):
        cp_dp_copies = split_cp_copies(copies)
        kv_read_cp_dp_copies.append(cp_dp_copies)
        kv_cache_reads[layer_id]._cp_dp_copies = cp_dp_copies

    barrier_copies = split_cp_copies(barrier_cp_copies)
    for layer in layers:
        layer._kv_persist_barrier_cp_dp_copies = barrier_copies

    _, emb_dp_copies, _ = g.split_kernel(dp_split, emb, dp)
    _, final_norm_dp_copies, _ = g.split_kernel(
        dp_split, final_norm, dp)
    _, logits_dp_copies, _ = g.split_kernel(dp_split, logits, dp)
    _, sampling_dp_copies, _ = g.split_kernel(dp_split, sampling, dp)

    prefix_fields = (
        "bridge", "attn_norm", "attn_fan", "wkv", "kv_norm",
    )
    for layer, cp_fields in zip(layers, cp_fields_by_layer):
        dp_fields = {}
        for name in prefix_fields:
            _, copies, _ = g.split_kernel(
                dp_split, getattr(layer, name), dp)
            dp_fields[name] = copies
            setattr(layer, f"_{name}_dp_copies", copies)
        layer._decode_dp_fields = dp_fields

        for name, copies in cp_fields.items():
            cp_dp_copies = split_cp_copies(copies)
            setattr(layer, f"_{name}_cp_dp_copies", cp_dp_copies)
        layer._kv_persist_fan_cp_dp_copies = (
            layer._kv_cache_fan_cp_dp_copies)

    canonicalize_split_comms(g, "dp")
    canonicalize_split_comms(g, "cp")
    canonicalize_split_comms(g, "dp")

    placement = Placement(hardware=hw, graph=g)

    def nearby_dram(device):
        node = device.name.split("-", 1)[0]
        gpu_index = int(device.name.rsplit("-", 1)[1])
        memories = drams[node]
        return memories[min(gpu_index * len(memories) // 8,
                            len(memories) - 1)]

    for layer_id, copies in enumerate(kv_read_cp_dp_copies):
        devices = stage_gpus[layer_stages[layer_id]]
        for rank, kernel in enumerate(copies):
            device = devices[rank]
            placement.set_kernel_device(kernel, device)
            placement.set_tensor_memory(
                kernel.inputs["kv"], nearby_dram(device))

    first_devices = stage_gpus[0]
    placement.set_kernel_device(read_input, first_devices[0])
    placement.set_tensor_memory(
        read_input.inputs["tokens"], nearby_dram(first_devices[0]))
    for dp_rank, kernel in enumerate(emb_dp_copies):
        placement.set_kernel_device(
            kernel, first_devices[dp_rank * cp])

    final_devices = stage_gpus[layer_stages[-1]]
    for copies in (
        final_norm_dp_copies,
        logits_dp_copies,
        sampling_dp_copies,
    ):
        for dp_rank, kernel in enumerate(copies):
            placement.set_kernel_device(
                kernel, final_devices[dp_rank * cp])

    for layer_id, layer in enumerate(layers):
        devices = stage_gpus[layer_stages[layer_id]]
        fields = layer._decode_dp_fields
        for name in prefix_fields:
            for dp_rank, kernel in enumerate(fields[name]):
                placement.set_kernel_device(
                    kernel, devices[dp_rank * cp])

        for name in (
            *_DECODE_REPLICATED_FIELDS,
            *_DECODE_SHARDED_FIELDS,
            "sa", "kv_cache_fan",
        ):
            copies = getattr(
                layer, f"_{name}_cp_dp_copies", None)
            if copies is None:
                continue
            for rank, kernel in enumerate(copies):
                placement.set_kernel_device(kernel, devices[rank])

        route_fields = {
            "dispatch": layer._dispatch_cp_dp_copies,
            "combine": layer._combine_cp_dp_copies,
        }
        _place_experts_and_routes(
            g, layer, route_fields, devices, placement, hw,
            shard_experts=True)

    for rank, barrier_copy in enumerate(barrier_copies):
        for input_name, tensor in barrier_copy.inputs.items():
            if input_name.startswith("kv"):
                layer_id = int(input_name[2:])
                devices = stage_gpus[layer_stages[layer_id]]
            else:
                devices = final_devices
            placement.set_tensor_memory(
                tensor, hw.find_local_memory(devices[rank]))
        output_memory = hw.find_local_memory(final_devices[rank])
        for tensor in barrier_copy.outputs.values():
            placement.set_tensor_memory(tensor, output_memory)

    # Restore direct cross-memory dependencies simplified before final
    # placement with explicit identity Moves.
    for kernel in list(g.topological_sort()):
        if isinstance(kernel, CommKernel):
            continue
        for edge in list(g._out_edges(kernel)):
            if isinstance(edge.dst, CommKernel):
                continue
            mismatched = {}
            tensors = []
            for output_name, input_name in edge.mapping.items():
                output_tensor = edge.src.outputs[output_name]
                input_tensor = edge.dst.inputs[input_name]
                source_memory = placement.get_tensor_memory(output_tensor)
                target_memory = placement.get_tensor_memory(input_tensor)
                if source_memory is target_memory:
                    continue
                mismatched[output_name] = input_name
                tensors.append(
                    (output_tensor, source_memory, target_memory))
            if not mismatched:
                continue
            move = Move()
            move.inputs = {
                f"src{index}": Tensor(tensor.dtype, tensor.shape)
                for index, (tensor, _, _) in enumerate(tensors)
            }
            move.outputs = {
                f"dst{index}": Tensor(tensor.dtype, tensor.shape)
                for index, (tensor, _, _) in enumerate(tensors)
            }
            g.insert_identity(move, edge.src, edge.dst, mismatched)
            source_device = placement.get_tensor_device(tensors[0][0])
            placement.set_kernel_device(move, source_device)
            for index, (_, source_memory, target_memory) in enumerate(tensors):
                placement.set_tensor_memory(
                    move.inputs[f"src{index}"], source_memory)
                placement.set_tensor_memory(
                    move.outputs[f"dst{index}"], target_memory)

    _place_comm_tensor_memories(g, placement)
    optimize_comms(g, placement)
    _place_comm_tensor_memories(g, placement)
    g.validate()
    placement.validate(g)
    return g, placement


def optimize_model_superchip(g, hw):
    """Place all kernels on the single fused GPU (no splits, no comms)."""
    gpu = [c for c in hw.nodes if isinstance(c, Compute)
           and "nvidia-b300" in c.name][0]
    p = Placement(hardware=hw, graph=g)
    for k in g.topological_sort():
        p.set_kernel_device(k, gpu)
    g.validate()
    p.validate(g)
    return g, p
