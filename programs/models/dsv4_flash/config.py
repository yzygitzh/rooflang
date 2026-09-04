# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""DeepSeek V4 Flash model configuration constants."""

D = 4096
N_LAYERS = 43
H = 64
HD = 512
Q_LORA = 1024
KV_DIM = 512
O_GROUPS = 8
O_LORA = 1024
N_EXPERTS = 256
TOPK = 6
MOE_INTER = 2048
WINDOW = 128
INDEX_TOPK = 512
INDEX_H = 64
INDEX_HD = 128
V = 129280
BATCH = 512
S_PREFILL = 8192
COMPRESS_RATIOS = [0, 0] + [v for _ in range(20) for v in (4, 128)] + [4]
