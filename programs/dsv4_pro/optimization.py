"""DeepSeek V4 Pro inference — graph splitting and placement strategies."""

from rooflang.language.hardware.component import Compute
from rooflang.language.optimization.comm import optimize_comms
from rooflang.language.optimization.split import batch_split, context_split
from rooflang.language.placement import Placement

from rooflang.programs.dsv4_pro.config import CP, DP, EP, N_LOCAL_EXPERTS


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

        # Expert kernels → respective GPUs
        # L.experts is flat: [up0, down0, up1, down1, ...] for N_EXPERTS experts
        # Group by GPU: expert eid → gpu_id = eid // N_LOCAL_EXPERTS
        for eid in range(N_LOCAL_EXPERTS * DP):
            gpu_id = eid // N_LOCAL_EXPERTS
            up_kernel = L.experts[eid * 2]
            down_kernel = L.experts[eid * 2 + 1]
            p.set_kernel_device(up_kernel, gpus[gpu_id])
            p.set_kernel_device(down_kernel, gpus[gpu_id])

        # Expert input locality: placed on destination GPU's HBM
        for eid in range(N_LOCAL_EXPERTS * DP):
            gpu_id = eid // N_LOCAL_EXPERTS
            local_mem = hw.find_local_memory(gpus[gpu_id])
            up_kernel = L.experts[eid * 2]
            p.set_tensor_memory(up_kernel.inputs["x"], local_mem)

        # Dispatch RDMA: each copy writes expert outputs to target GPU's HBM
        for copy in L._dispatch_copies:
            for gpu_id in range(DP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.outputs[f"o{global_eid}"], local_mem)

        # Combine RDMA: each copy reads expert outputs from source GPU's HBM
        for copy in L._combine_copies:
            for gpu_id in range(DP):
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

    gpus, cpu = _node_resources(hw, 0, CP)
    if EP != len(gpus):
        raise ValueError(
            f"CP8/EP8 placement requires {EP} colocated EP ranks; "
            f"found {len(gpus)} GPUs")

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

    optimize_comms(g, placement)
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
