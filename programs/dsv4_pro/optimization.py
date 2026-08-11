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
    decode_attention_context_split, decode_persistence_split,
    replicate_before,
)
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor

from rooflang.programs.dsv4_pro.config import (
    COMPRESS_RATIOS, CP, DP, EP, N_EXPERTS, N_LOCAL_EXPERTS, TOPK, WINDOW,
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


def _node_resources(hw, node_id, world):
    """Return the eight B300 GPUs and first CPU belonging to one node."""
    prefix = f"n{node_id}-"
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and c.name.startswith(prefix) and "nvidia-b300" in c.name],
        key=lambda c: c.name)
    cpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and c.name.startswith(prefix) and "intel-xeon" in c.name],
        key=lambda c: c.name)
    if len(gpus) != world or not cpus:
        raise ValueError(
            f"B300 node {node_id} requires {world} GPUs and at least one CPU; "
            f"found {len(gpus)} GPUs and {len(cpus)} CPUs")
    return gpus, cpus[0]


def _optimize_model_b300_cluster_a_dp8_ep8(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None,
    prefill_node=0, decode_node=0,
):
    """Apply cluster DP/EP splits and place each phase on its node."""
    if DP != EP:
        raise ValueError(
            f"DP/EP placement requires DP == EP; got DP={DP}, EP={EP}")

    prefill_gpus, prefill_cpu = _node_resources(hw, prefill_node, DP)
    decode_gpus, decode_cpu = _node_resources(hw, decode_node, DP)

    # ── Phase 1: DP splits (batch dim) ───────────────────────────────
    emb_copies = None
    if emb is not None:
        _, emb_copies, _ = g.split_kernel(batch_split, emb, DP)

    def _split_layer(L):
        """Split all kernels in a LayerMeta by batch dimension."""
        _, L._bridge_copies, _ = g.split_kernel(batch_split, L.bridge, DP)
        _, L._attn_norm_copies, _ = g.split_kernel(batch_split, L.attn_norm, DP)
        _, L._attn_fan_copies, _ = g.split_kernel(batch_split, L.attn_fan, DP)
        if L.comp is not None:
            _, L._comp_copies, _ = g.split_kernel(batch_split, L.comp, DP)
        if L.comp_buf is not None:
            _, L._comp_buf_copies, _ = g.split_kernel(
                batch_split, L.comp_buf, DP)
        if L.comp_concat is not None:
            _, L._comp_concat_copies, _ = g.split_kernel(
                batch_split, L.comp_concat, DP)
        if L.comp_norm is not None:
            _, L._comp_norm_copies, _ = g.split_kernel(
                batch_split, L.comp_norm, DP)
        _, L._wq_a_copies, _ = g.split_kernel(batch_split, L.wq_a, DP)
        _, L._q_norm_copies, _ = g.split_kernel(batch_split, L.q_norm, DP)
        _, L._wq_b_copies, _ = g.split_kernel(batch_split, L.wq_b, DP)
        _, L._wkv_copies, _ = g.split_kernel(batch_split, L.wkv, DP)
        _, L._kv_norm_copies, _ = g.split_kernel(batch_split, L.kv_norm, DP)
        if L.kv_norm_fan is not None:
            _, L._kv_norm_fan_copies, _ = g.split_kernel(
                batch_split, L.kv_norm_fan, DP)
        if L.comp_norm_fan is not None:
            _, L._comp_norm_fan_copies, _ = g.split_kernel(
                batch_split, L.comp_norm_fan, DP)
        if L.kv_concat is not None:
            _, L._kv_concat_copies, _ = g.split_kernel(
                batch_split, L.kv_concat, DP)
        _, L._sa_copies, _ = g.split_kernel(batch_split, L.sa, DP)
        _, L._wo_a_copies, _ = g.split_kernel(batch_split, L.wo_a, DP)
        _, L._wo_b_copies, _ = g.split_kernel(batch_split, L.wo_b, DP)
        _, L._attn_add_copies, _ = g.split_kernel(batch_split, L.attn_add, DP)
        _, L._ffn_bridge_copies, _ = g.split_kernel(
            batch_split, L.ffn_bridge, DP)
        _, L._ffn_norm_copies, _ = g.split_kernel(batch_split, L.ffn_norm, DP)
        _, L._ffn_fan_copies, _ = g.split_kernel(batch_split, L.ffn_fan, DP)
        _, L._gate_copies, _ = g.split_kernel(batch_split, L.gate, DP)
        _, L._dispatch_copies, _ = g.split_kernel(batch_split, L.dispatch, DP)
        _, L._combine_copies, _ = g.split_kernel(
            batch_split, L.combine, DP)
        _, L._sw_up_copies, _ = g.split_kernel(batch_split, L.sw_up, DP)
        _, L._sw_down_copies, _ = g.split_kernel(batch_split, L.sw_down, DP)
        _, L._moe_add_copies, _ = g.split_kernel(batch_split, L.moe_add, DP)
        _, L._ffn_add_copies, _ = g.split_kernel(batch_split, L.ffn_add, DP)
        if L.kv_acc is not None:
            _, L._kv_acc_copies, _ = g.split_kernel(
                batch_split, L.kv_acc, DP)
        if L.kv_win_slice is not None:
            _, L._kv_win_slice_copies, _ = g.split_kernel(
                batch_split, L.kv_win_slice, DP)
        if L.kv_cache_fan is not None:
            _, L._kv_cache_fan_copies, _ = g.split_kernel(
                batch_split, L.kv_cache_fan, DP)
        if L.kv_comp_slice is not None:
            _, L._kv_comp_slice_copies, _ = g.split_kernel(
                batch_split, L.kv_comp_slice, DP)
        if L.kv_acc_fan is not None:
            _, L._kv_acc_fan_copies, _ = g.split_kernel(
                batch_split, L.kv_acc_fan, DP)

    for L in layers:
        _split_layer(L)

    # Split decode steps
    if decode_steps:
        for decode_step in decode_steps:
            if decode_step.emb is not None:
                _, decode_step._emb_copies, _ = g.split_kernel(
                    batch_split, decode_step.emb, DP)
            _, decode_step._final_norm_copies, _ = g.split_kernel(
                batch_split, decode_step.final_norm, DP)
            _, decode_step._logits_copies, _ = g.split_kernel(
                batch_split, decode_step.logits, DP)
            _, decode_step._sampling_copies, _ = g.split_kernel(
                batch_split, decode_step.sampling, DP)
            for L in decode_step.layers:
                _split_layer(L)

    # Split prefill output head
    if prefill_output_head:
        _pfx_out_copies = []
        for k in prefill_output_head:
            _, copies, _ = g.split_kernel(batch_split, k, DP)
            _pfx_out_copies.append(copies)

    # Split KV cache reads (decode-only)
    if kv_cache_reads:
        _kv_read_copies = []
        for kv_read in kv_cache_reads:
            _, copies, _ = g.split_kernel(batch_split, kv_read, DP)
            _kv_read_copies.append(copies)

    # ── Placement ─────────────────────────────────────────────────
    p = Placement(hardware=hw, graph=g)

    if emb_copies is not None:
        for i, c in enumerate(emb_copies):
            p.set_kernel_device(c, prefill_gpus[i])

    if read_input is not None:
        p.set_kernel_device(read_input, prefill_gpus[0])
        prefill_cpu_mem = hw.find_local_memory(prefill_cpu)
        p.set_tensor_memory(read_input.inputs["tokens"], prefill_cpu_mem)

    def _place_layer(L, gpus):
        """Place all DP copies of a layer onto their respective GPUs."""
        always_copies = [L._bridge_copies,
                         L._attn_norm_copies,
                         L._attn_fan_copies, L._wq_a_copies,
                         L._q_norm_copies, L._wq_b_copies,
                         L._wkv_copies, L._kv_norm_copies,
                         L._sa_copies,
                         L._wo_a_copies, L._wo_b_copies,
                         L._attn_add_copies, L._ffn_bridge_copies,
                         L._ffn_norm_copies, L._ffn_fan_copies,
                         L._gate_copies, L._dispatch_copies,
                         L._combine_copies,
                         L._sw_up_copies, L._sw_down_copies,
                         L._moe_add_copies, L._ffn_add_copies]
        if L.kv_concat is not None:
            always_copies.append(L._kv_concat_copies)
        if L.kv_norm_fan is not None:
            always_copies.append(L._kv_norm_fan_copies)
        if L.comp_norm_fan is not None:
            always_copies.append(L._comp_norm_fan_copies)
        if L.kv_acc is not None:
            always_copies.append(L._kv_acc_copies)
        if L.kv_win_slice is not None:
            always_copies.append(L._kv_win_slice_copies)
        if L.kv_cache_fan is not None:
            always_copies.append(L._kv_cache_fan_copies)
        if L.kv_comp_slice is not None:
            always_copies.append(L._kv_comp_slice_copies)
        if L.kv_acc_fan is not None:
            always_copies.append(L._kv_acc_fan_copies)
        for copies in always_copies:
            for i, c in enumerate(copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp is not None:
            for i, c in enumerate(L._comp_copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp_buf is not None:
            for i, c in enumerate(L._comp_buf_copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp_concat is not None:
            for i, c in enumerate(L._comp_concat_copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp_norm is not None:
            for i, c in enumerate(L._comp_norm_copies):
                p.set_kernel_device(c, gpus[i])

        # Expert kernels are EP-sharded, not copied by batch split.
        # L.experts is flat: [up0, down0, up1, down1, ...] for N_EXPERTS experts
        # Group by GPU: expert eid → gpu_id = eid // N_LOCAL_EXPERTS
        for eid in range(N_LOCAL_EXPERTS * EP):
            gpu_id = eid // N_LOCAL_EXPERTS
            up_kernel = L.experts[eid * 2]
            down_kernel = L.experts[eid * 2 + 1]
            p.set_kernel_device(up_kernel, gpus[gpu_id])
            p.set_kernel_device(down_kernel, gpus[gpu_id])

        # Expert input locality: placed on destination GPU's HBM
        for eid in range(N_LOCAL_EXPERTS * EP):
            gpu_id = eid // N_LOCAL_EXPERTS
            local_mem = hw.find_local_memory(gpus[gpu_id])
            up_kernel = L.experts[eid * 2]
            p.set_tensor_memory(up_kernel.inputs["x"], local_mem)

        # Dispatch RDMA: each copy writes expert outputs to target GPU's HBM
        for copy in L._dispatch_copies:
            for gpu_id in range(EP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.outputs[f"o{global_eid}"], local_mem)

        # Combine RDMA: each copy reads expert outputs from source GPU's HBM
        for copy in L._combine_copies:
            for gpu_id in range(EP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.inputs[f"i{global_eid}"], local_mem)

    for L in layers:
        _place_layer(L, prefill_gpus)

    # Place decode steps
    if decode_steps:
        decode_cpu_mem = hw.find_local_memory(decode_cpu)
        for decode_step in decode_steps:
            if decode_step.read_input is not None:
                p.set_kernel_device(decode_step.read_input, decode_gpus[0])
                p.set_tensor_memory(
                    decode_step.read_input.inputs["tokens"], decode_cpu_mem)
            if decode_step.emb is not None:
                for i, c in enumerate(decode_step._emb_copies):
                    p.set_kernel_device(c, decode_gpus[i])
            for i, c in enumerate(decode_step._final_norm_copies):
                p.set_kernel_device(c, decode_gpus[i])
            for i, c in enumerate(decode_step._logits_copies):
                p.set_kernel_device(c, decode_gpus[i])
            for i, c in enumerate(decode_step._sampling_copies):
                p.set_kernel_device(c, decode_gpus[i])
            for L in decode_step.layers:
                _place_layer(L, decode_gpus)

    # Place prefill output head
    if prefill_output_head:
        for layer_copies in _pfx_out_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, prefill_gpus[i])

    # Place KV cache reads (decode-only)
    if kv_cache_reads:
        decode_cpu_mem = hw.find_local_memory(decode_cpu)
        for layer_copies in _kv_read_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, decode_gpus[i])
                p.set_tensor_memory(c.inputs["kv"], decode_cpu_mem)

    _place_comm_tensor_memories(g, p)
    optimize_comms(g, p)

    g.validate()
    p.validate(g)
    return g, p


def optimize_model_b300_cluster_a_dp8_ep8_1node(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None,
):
    """Place both prefill and decode on node 0 of B300 Cluster A."""
    return _optimize_model_b300_cluster_a_dp8_ep8(
        g, layers, hw, emb, read_input, decode_steps, kv_cache_reads,
        prefill_output_head, prefill_node=0, decode_node=0)


def optimize_model_b300_cluster_a_dp8_ep8_2node(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None,
):
    """Place prefill on node 0 and decode on node 1."""
    return _optimize_model_b300_cluster_a_dp8_ep8(
        g, layers, hw, emb, read_input, decode_steps, kv_cache_reads,
        prefill_output_head, prefill_node=0, decode_node=1)


def optimize_model_b300_cluster_a_cp8_ep8_1node(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None,
):
    """Ring CP=8 plus EP=8 prefill on one B300 Cluster A node.

    Decode is intentionally unsupported: its sequence dimension is one and
    cannot be split across CP ranks.
    """
    if decode_steps or kv_cache_reads or prefill_output_head:
        raise ValueError(
            "optimize_model_b300_cluster_a_cp8_ep8_1node supports "
            "prefill only; decode sequence length cannot be CP-sharded")

    if CP != EP:
        raise ValueError(
            f"CP/EP placement requires CP == EP; got CP={CP}, EP={EP}")

    gpus, cpu = _node_resources(hw, 0, CP)

    emb_copies = None
    if emb is not None:
        _, emb_copies, _ = g.split_kernel(context_split, emb, CP)

    layer_fields = (
        "bridge", "attn_norm", "attn_fan", "comp", "comp_norm",
        "wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "kv_concat",
        "sa", "kv_win_slice", "wo_a", "wo_b", "attn_add",
        "ffn_bridge", "ffn_norm", "ffn_fan", "gate", "dispatch",
        "combine", "sw_up", "sw_down", "moe_add", "ffn_add",
    )
    for layer in layers:
        for name in layer_fields:
            kernel = getattr(layer, name)
            if kernel is None:
                continue
            _, copies, _ = g.split_kernel(context_split, kernel, CP)
            setattr(layer, f"_{name}_copies", copies)

    placement = Placement(hardware=hw, graph=g)
    if emb_copies is not None:
        for rank, copy in enumerate(emb_copies):
            placement.set_kernel_device(copy, gpus[rank])

    if read_input is not None:
        placement.set_kernel_device(read_input, gpus[0])
        placement.set_tensor_memory(
            read_input.inputs["tokens"],
            hw.find_local_memory(cpu))

    for layer in layers:
        for name in layer_fields:
            copies = getattr(layer, f"_{name}_copies", None)
            if copies is None:
                continue
            for rank, copy in enumerate(copies):
                placement.set_kernel_device(copy, gpus[rank])

        # Expert kernels are EP-sharded, not copied by context split.
        for eid in range(N_LOCAL_EXPERTS * EP):
            gpu_id = eid // N_LOCAL_EXPERTS
            up_kernel = layer.experts[eid * 2]
            down_kernel = layer.experts[eid * 2 + 1]
            placement.set_kernel_device(up_kernel, gpus[gpu_id])
            placement.set_kernel_device(down_kernel, gpus[gpu_id])

            local_mem = hw.find_local_memory(gpus[gpu_id])
            placement.set_tensor_memory(up_kernel.inputs["x"], local_mem)

        # Dispatch writes and Combine reads expert shards in the owner rank.
        for copy in layer._dispatch_copies:
            for gpu_id in range(EP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    placement.set_tensor_memory(
                        copy.outputs[f"o{global_eid}"], local_mem)

        for copy in layer._combine_copies:
            for gpu_id in range(EP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    placement.set_tensor_memory(
                        copy.inputs[f"i{global_eid}"], local_mem)

    _place_comm_tensor_memories(g, placement)
    optimize_comms(g, placement)
    g.validate()
    placement.validate(g)
    return g, placement


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
    layers, emb, cp, dp, ep, pp, n_gpus,
):
    """Validate parallel degrees, PP layer counts, and split divisibility."""
    if min(cp, dp, ep, n_gpus) <= 0:
        raise ValueError("cp, dp, ep, and n_gpus must all be positive")
    if not pp or any(not isinstance(count, int) or count <= 0 for count in pp):
        raise ValueError("pp layer counts must be positive integers")
    if sum(pp) != len(layers):
        raise ValueError(
            f"pp layer counts must sum to the model layer count; "
            f"got sum(pp)={sum(pp)} for {len(layers)} layers")
    pp_degree = len(pp)
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


def optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None, *,
    cp, dp, ep, pp, n_gpus,
):
    """Apply dynamic DP→CP→EP→PP placement to a prefill-only graph.

    Non-expert kernels use CP×DP within their assigned PP stage. Experts use
    EP within the stage, with EP=CP×DP. Each pp entry is the number of model
    layers assigned to that stage. PP models one prefill wave without a
    microbatch schedule.
    """
    if decode_steps or kv_cache_reads or prefill_output_head:
        raise ValueError(
            "optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill supports "
            "prefill only")
    _validate_cp_dp_ep_pp_prefill(
        layers, emb, cp, dp, ep, pp, n_gpus)
    pp_degree = len(pp)
    layer_stages = [
        stage
        for stage, layer_count in enumerate(pp)
        for _ in range(layer_count)
    ]
    gpus, cpus, drams = _cluster_a_resources(hw)
    if len(gpus) != n_gpus:
        raise ValueError(
            f"hardware contains {len(gpus)} B300 GPUs, expected {n_gpus}")
    if not cpus or not drams:
        raise ValueError("B300 Cluster A requires per-node CPUs and DRAMs")

    kv_barrier = layers[0].kv_persist_barrier
    if not all(layer.persist_kv_cache for layer in layers) \
            or kv_barrier is None \
            or any(layer.kv_persist_barrier is not kv_barrier
                   for layer in layers):
        raise ValueError(
            "prefill KV persistence is disabled; pass "
            "persist_kv_cache=True to declare_model")

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
        _, split_copies, _ = g.split_kernel(context_split, copy, cp)
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


def optimize_model_b300_cluster_a_cp_dp_ep_pp_decode(
    g, layers, hw, emb=None, read_input=None, decode_steps=None,
    kv_cache_reads=None, prefill_output_head=None, *,
    cp, dp, ep, pp, n_gpus,
):
    """Apply CP, DP, EP, and PP to a one-step decode-only graph.

    KV is a persistent read-only model input. The optimizer does not construct
    cache slices, append the current token, or model cache ownership changes.
    """
    if layers or emb is not None or read_input is not None \
            or prefill_output_head:
        raise ValueError(
            "optimize_model_b300_cluster_a_cp_dp_ep_pp_decode supports "
            "decode only")
    if not decode_steps or not kv_cache_reads:
        raise ValueError("decode-only optimizer requires decode steps and KV")
    if len(decode_steps) != 1:
        raise ValueError("decode optimizer supports exactly one step")
    if min(cp, dp, ep, n_gpus) <= 0:
        raise ValueError("cp, dp, ep, and n_gpus must all be positive")
    if not pp or any(not isinstance(count, int) or count <= 0 for count in pp):
        raise ValueError("pp layer counts must be positive integers")

    step = decode_steps[0]
    n_layers = len(step.layers)
    if sum(pp) != n_layers:
        raise ValueError(
            "pp layer counts must sum to the model layer count; "
            f"got sum(pp)={sum(pp)} for {n_layers} layers")
    pp_degree = len(pp)
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

    batch_size = step.emb.outputs["y"].shape[0]
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
        for stage, layer_count in enumerate(pp)
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
            zip(kv_cache_reads, step.layers)):
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

    barrier = step.kv_persist_barrier
    if barrier is None or any(
            layer.kv_persist_barrier is not barrier
            for layer in step.layers):
        raise ValueError("decode KV persistence barrier is missing")
    barrier_prev, barrier_cp_copies, _ = g.split_kernel(
        decode_persistence_split, barrier, cp)
    cp_collectives_to_replicate = [barrier_prev["decode_output"]]

    cp_fields_by_layer = []
    for layer in step.layers:
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
    step._kv_persist_barrier_cp_dp_copies = barrier_copies
    for layer in step.layers:
        layer._kv_persist_barrier_cp_dp_copies = barrier_copies

    _, step._emb_dp_copies, _ = g.split_kernel(dp_split, step.emb, dp)
    _, step._final_norm_dp_copies, _ = g.split_kernel(
        dp_split, step.final_norm, dp)
    _, step._logits_dp_copies, _ = g.split_kernel(
        dp_split, step.logits, dp)
    _, step._sampling_dp_copies, _ = g.split_kernel(
        dp_split, step.sampling, dp)

    prefix_fields = (
        "bridge", "attn_norm", "attn_fan", "wkv", "kv_norm",
    )
    for layer, cp_fields in zip(step.layers, cp_fields_by_layer):
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
    if step.read_input is not None:
        placement.set_kernel_device(step.read_input, first_devices[0])
        placement.set_tensor_memory(
            step.read_input.inputs["tokens"], nearby_dram(first_devices[0]))
    for dp_rank, kernel in enumerate(step._emb_dp_copies):
        placement.set_kernel_device(
            kernel, first_devices[dp_rank * cp])

    final_devices = stage_gpus[layer_stages[-1]]
    for copies in (
        step._final_norm_dp_copies,
        step._logits_dp_copies,
        step._sampling_dp_copies,
    ):
        for dp_rank, kernel in enumerate(copies):
            placement.set_kernel_device(
                kernel, final_devices[dp_rank * cp])

    for layer_id, layer in enumerate(step.layers):
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

def optimize_model_b300_superchip_a(g, hw):
    """Place all kernels on the single fused GPU (no splits, no comms)."""
    gpu = [c for c in hw.nodes if isinstance(c, Compute)
           and "nvidia-b300" in c.name][0]
    p = Placement(hardware=hw, graph=g)
    for k in g.topological_sort():
        p.set_kernel_device(k, gpu)
    g.validate()
    p.validate(g)
    return g, p
