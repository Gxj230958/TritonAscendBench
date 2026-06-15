---
title: TAB shape-extension pilot dry-run (Unit 2, CPU-validated candidates, PENDING GPU)
date: 2026-06-15
tool: scripts/corpus/extend_task_shapes.py --dry-run
---

# Shape-extension pilot dry-run (5 small-ref overlap tasks)

Every candidate below is **CPU-valid (materializable + oracle-domain-valid + constraint-clean)**
but **NOT proven kernel-valid** -- a frozen Triton kernel can still reject it (e.g. a feature-dim
BLOCK_N cap). They are labelled `CPU_VALID_PENDING_GPU`; Unit 3 (GPU S2/C2 re-verify) is what
PROMOTES or QUARANTINES each. Existing P1-P5/H1-H4 are byte-preserved (append-only). No shape has
been applied yet -- this is the dry-run proof the tool produces valid candidates.

## L2_01_layer_norm_fwd
```
active knobs (doc order): ['B', 'D']
survivors (CPU-valid, PENDING GPU confirmation): 10 (8 public + 2 hidden)
  [ CPU_VALID_PENDING_GPU] borrow   P6   {'B': 128, 'D': 1536}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P7   {'B': 128, 'D': 6144}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P8   {'B': 512, 'D': 3072}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P9   {'B': 1000, 'D': 4096}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P10  {'B': 2048, 'D': 3584}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] diagonal P11  {'B': 1, 'D': 16384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] diagonal P12  {'B': 192, 'D': 12288}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] diagonal P13  {'B': 256, 'D': 16384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H5   {'B': 1, 'D': 2}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H6   {'B': 1, 'D': 1024}  -- CPU-valid; pending GPU kernel confirmation
```

## L2_04_silu_and_mul
```
active knobs (doc order): ['B', 'D']
survivors (CPU-valid, PENDING GPU confirmation): 5 (3 public + 2 hidden)
  [ CPU_VALID_PENDING_GPU] diagonal P6   {'B': 1, 'D': 32768}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P7   {'B': 6144, 'D': 1024}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P8   {'B': 8192, 'D': 1024}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H5   {'B': 1, 'D': 4}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H6   {'B': 1, 'D': 1024}  -- CPU-valid; pending GPU kernel confirmation
```

## L2_05_gelu_and_mul
```
active knobs (doc order): ['B', 'D', 'N']
survivors (CPU-valid, PENDING GPU confirmation): 3 (1 public + 2 hidden)
  [ CPU_VALID_PENDING_GPU] diagonal P6   {'B': 1, 'D': 1024, 'N': 8192}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H5   {'B': 1, 'D': 4, 'N': 1}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H6   {'B': 1, 'D': 1024, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
```

## L2_20_bmm
```
active knobs (doc order): ['B', 'L', 'D', 'N']
survivors (CPU-valid, PENDING GPU confirmation): 9 (7 public + 2 hidden)
  [ CPU_VALID_PENDING_GPU] borrow   P6   {'B': 1, 'L': 2, 'D': 4, 'N': 2}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P7   {'B': 2, 'L': 16, 'D': 32, 'N': 16}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P8   {'B': 2, 'L': 32, 'D': 64, 'N': 32}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P9   {'B': 2, 'L': 64, 'D': 128, 'N': 64}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P10  {'B': 3, 'L': 6, 'D': 12, 'N': 6}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] borrow   P11  {'B': 8, 'L': 64, 'D': 128, 'N': 64}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] diagonal P12  {'B': 1, 'L': 128, 'D': 512, 'N': 8192}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H5   {'B': 1, 'L': 1, 'D': 1, 'N': 1}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H6   {'B': 1, 'L': 128, 'D': 512, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
```

## L2_21_matmul_kernel_persistent
```
active knobs (doc order): ['B', 'D', 'N']
survivors (CPU-valid, PENDING GPU confirmation): 9 (7 public + 2 hidden)
  [ CPU_VALID_PENDING_GPU] diagonal P6   {'B': 1, 'D': 512, 'N': 8192}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P7   {'B': 64, 'D': 512, 'N': 6144}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P8   {'B': 64, 'D': 512, 'N': 8192}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P9   {'B': 64, 'D': 6144, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P10  {'B': 64, 'D': 8192, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P11  {'B': 1536, 'D': 512, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] scale    P12  {'B': 2048, 'D': 512, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H5   {'B': 1, 'D': 1, 'N': 1}  -- CPU-valid; pending GPU kernel confirmation
  [ CPU_VALID_PENDING_GPU] edge     H6   {'B': 1, 'D': 512, 'N': 384}  -- CPU-valid; pending GPU kernel confirmation
```

