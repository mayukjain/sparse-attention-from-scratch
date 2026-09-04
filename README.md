# Sparse Attention from Scratch

A minimal GPT-style transformer built from scratch and extended with sparse attention variants. Baseline follows Karpathy's Zero to Hero series; sparse extensions explore how attention patterns affect quality vs. compute tradeoffs on TinyShakespeare.

## Motivation

Standard dense attention scales as O(n²) in sequence length. This project implements two sparse alternatives — sliding window and BigBird-style (sliding + global + random) — and compares them against the dense baseline on training quality, inference latency, and memory usage.

## Current state

- [x] Working dense transformer (val loss ~1.53 at 2500 iters on TinyShakespeare)
- [x] Sliding window attention
- [x] BigBird-style attention (sliding + global + random)
- [ ] Correctness harness (partial — mask function tests done, more coverage planned)
- [x] Numerical stability handling for fully-masked rows
- [x] Benchmark: dense vs. sparse across sequence lengths (512 → 8192)
- [ ] Quality comparison on TinyShakespeare (in progress)
- [ ] Writeup

## Results

### Benchmark: forward pass time and memory vs sequence length

![Benchmark plot](assets/benchmark_plot.png)

| Variant  | Seq 512 | Seq 1024 | Seq 2048 | Seq 4096 | Seq 8192 |
|----------|---------|----------|----------|----------|----------|
| Dense    | 0.22 ms | 0.69 ms  | 2.37 ms  | 9.13 ms  | 35.15 ms |
| Sliding  | 0.18 ms | 0.60 ms  | 2.18 ms  | 9.14 ms  | 35.18 ms |
| BigBird  | 0.18 ms | 0.61 ms  | 2.19 ms  | 9.13 ms  | 35.16 ms |

Peak memory is identical across all three variants at each sequence length (13.2 MB → 924.4 MB).

**Finding:** naive mask-based sparse attention exhibits identical time and memory scaling to dense (O(n²)). This is expected — `masked_fill` zeroes entries in an already-allocated T×T score matrix, so compute and memory don't decrease. Real sparse attention (e.g., Longformer's CUDA implementation) avoids materializing the full matrix; that would require custom kernel code beyond this task's scope.

Hardware: RTX 3050 6GB, batch size 1, single attention head.

## Setup

```bash
conda activate ml
python nano.py
```

Requires: PyTorch (CUDA optional but recommended), Python 3.10+.

## References

- Karpathy, *Let's Build GPT* — architectural foundation.
- Beltagy et al., *Longformer* (2020) — sliding window attention.
- Zaheer et al., *Big Bird* (2020) — block-sparse + global + random patterns.