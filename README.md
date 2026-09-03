# Sparse Attention from Scratch

A minimal GPT-style transformer built from scratch and extended with sparse attention variants. Baseline follows Karpathy's Zero to Hero series; sparse extensions explore how attention patterns affect quality vs. compute tradeoffs on TinyShakespeare.

## Motivation

Standard dense attention scales as O(n²) in sequence length. This project implements two sparse alternatives — sliding window and block-sparse — and compares them against the dense baseline on training quality, inference latency, and memory usage.

## Current state

- [x] Working dense transformer (val loss ~1.53 at 2500 iters on TinyShakespeare)
- [ ] Sliding window attention
- [ ] Block-sparse attention  
- [ ] Correctness harness (sparse ≡ dense when masks align)
- [ ] Numerical stability handling for fully-masked rows
- [ ] Benchmark: dense vs. sparse across sequence lengths
- [ ] Quality comparison on TinyShakespeare
- [ ] Writeup

## Setup

```bash
conda activate ml
python nano.py
```

Requires: PyTorch (CUDA optional but recommended), Python 3.10+.

## References

- Karpathy, *Let's Build GPT* — architectural foundation.
- Beltagy et al., *Longformer* (2020) — sliding window attention.
- Zaheer et al., *Big Bird* (2020) — block-sparse patterns.
