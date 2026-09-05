# Sparse Attention from Scratch

A minimal GPT-style transformer built from scratch and extended with sparse attention variants. Baseline follows Karpathy's Zero to Hero series; sparse extensions explore how attention patterns affect quality vs. compute tradeoffs on TinyShakespeare.

## Motivation

Standard dense attention scales as O(n²) in sequence length. This project implements two sparse alternatives — sliding window and BigBird-style (sliding + global + random) — and compares them against the dense baseline on training quality, inference latency, and memory usage.

## Current state

- [x] Working dense transformer (val loss ~1.53 at 2500 iters on TinyShakespeare)
- [x] Sliding window attention(window_size=128, val loss ~1.53 at 2500 iters)
- [x] BigBird-style attention (sliding + global + random; window=128, global=2, random=3; val loss ~1.52 at 2500 iters)
- [x] Correctness harness (8 tests covering sliding + bigbird structure, causality, boundaries)
- [x] Numerical stability handling for fully-masked rows
- [x] Benchmark: dense vs. sparse across sequence lengths (512 → 8192)
- [x] Quality comparison on TinyShakespeare (in progress)
- [x] Writeup (see [WRITEUP.md](WRITEUP.md))
- [ ] future expansions (custom kernal big bird)

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
### Quality comparison on TinyShakespeare

Trained with matched hyperparameters (block_size=256, n_layer=2, n_head=6, batch_size=64, 5000 iterations).

| Variant  | Config                              | Val Loss |
|----------|-------------------------------------|----------|
| Dense    | full causal                         | 1.5674   |
| Sliding  | window_size=128                     | 1.5646   |
| BigBird  | window=128, global=2, random=3      | 1.5490   |

BigBird outperforms both dense and sliding on this task by a small but consistent margin. See [WRITEUP.md](WRITEUP.md) for full discussion.

Generation samples from each trained model: [samples.txt](samples.txt).

## Setup

Prerequisites: PyTorch (CUDA optional but recommended), Python 3.10+, matplotlib.

Training a model:
```bash
conda activate ml
python nano.py
```

Configure the attention type in `hyperparams.py` by setting `mask_type` to `'dense'`, `'sliding'`, or `'bigbird'`.

Running the correctness harness:
```bash
python harness.py
```

This will produce `benchmark_results.csv` and `assets/benchmark_plot.png`.

## References

[1] Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.  
[2] Beltagy, Peters, & Cohan (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.  
[3] Zaheer et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS.  
[4] Dao et al. (2022). *FlashAttention*. NeurIPS.
