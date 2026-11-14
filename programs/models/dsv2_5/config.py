# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""DeepSeek V2.5 model configuration constants.

Values follow the official ``deepseek-ai/DeepSeek-V2.5`` config.json: 60
decoder layers (layer 0 dense, layers 1-59 MoE), 236B total / 21B active
parameters, MLA with a 512+64 latent KV cache, and 160 routed + 2 shared
experts.  The checkpoint is fully BF16.

Routing uses group-limited greedy selection (n_group=8, topk_group=3).  The
group-selection cost is negligible next to the expert GEMMs, so it is not
modeled as a separate kernel.
"""

D = 5120
N_LAYERS = 60
H = 128
Q_LORA = 1536
KV_LORA = 512
QK_NOPE_HD = 128
QK_ROPE_HD = 64
QK_HD = QK_NOPE_HD + QK_ROPE_HD
V_HD = 128
KV_CACHE_DIM = KV_LORA + QK_ROPE_HD
DENSE_LAYERS = 1
DENSE_INTER = 12288
N_EXPERTS = 160
TOPK = 6
MOE_INTER = 1536
N_SHARED_EXPERTS = 2
SHARED_INTER = N_SHARED_EXPERTS * MOE_INTER
ROUTER_SCORING_FUNC = "softmax"
V = 102400
BATCH = 512
S_PREFILL = 8192
