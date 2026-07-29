"""DeepSeek V4 Pro inference — Optimization Phase.

Applies split_kernel for DP, control edges, and placement.
"""

from rooflang.language.hardware.component import Compute
from rooflang.language.optimization.comm import optimize_comms
from rooflang.language.optimization.split import batch_split
from rooflang.language.placement import Placement

from rooflang.programs.dsv4_pro.config import DP, N_LOCAL_EXPERTS


def optimize_model(g, layers, hw, emb=None, read_input=None,
                   decode_steps=None, kv_cache_reads=None,
                   prefill_output_head=None):
    """Apply split_kernel for DP, add control edges, and place."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and "nvidia-b300" in c.name],
        key=lambda c: c.name)

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
            p.set_kernel_device(c, gpus[i])

    if read_input is not None:
        p.set_kernel_device(read_input, gpus[0])
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        p.set_tensor_memory(read_input.inputs["tokens"], cpu_mem)

    def _place_layer(L):
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
        _place_layer(L)

    # Place decode steps
    if decode_steps:
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        for decode_step in decode_steps:
            if decode_step.read_input is not None:
                p.set_kernel_device(decode_step.read_input, gpus[0])
                p.set_tensor_memory(
                    decode_step.read_input.inputs["tokens"], cpu_mem)
            if decode_step.emb is not None:
                for i, c in enumerate(decode_step._emb_copies):
                    p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(decode_step._final_norm_copies):
                p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(decode_step._logits_copies):
                p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(decode_step._sampling_copies):
                p.set_kernel_device(c, gpus[i])
            for L in decode_step.layers:
                _place_layer(L)

    # Place prefill output head
    if prefill_output_head:
        for layer_copies in _pfx_out_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, gpus[i])

    # Place KV cache reads (decode-only)
    if kv_cache_reads:
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        for layer_copies in _kv_read_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, gpus[i])
                p.set_tensor_memory(c.inputs["kv"], cpu_mem)

    optimize_comms(g, p)

    g.validate()
    p.validate(g)
    return g, p


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
