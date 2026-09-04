# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""DeepSeek V4 Pro model configuration constants."""

D = 7168
N_LAYERS = 61
H = 128
HD = 512
Q_LORA = 1536
KV_DIM = 512
O_GROUPS = 16
O_LORA = 1024
N_EXPERTS = 384
TOPK = 6
MOE_INTER = 3072
WINDOW = 128
INDEX_TOPK = 1024
INDEX_H = 64
INDEX_HD = 128
V = 129280
BATCH = 512
S_PREFILL = 8192
COMPRESS_RATIOS = [128, 128] + [v for _ in range(29) for v in (4, 128)] + [4]
