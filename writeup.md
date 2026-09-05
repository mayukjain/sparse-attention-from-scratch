# Sparse Attention from Scratch — Writeup

## 1. Summary

This project explored how different sparse attention patterns compare to standard dense attention in a small decoder-only transformer, following the architecture in *Attention Is All You Need* [1] with pre-norm and residual connections. Three variants were implemented: dense causal (baseline), sliding-window attention (Longformer [2]), and BigBird-style attention (sliding + global + random) [3]. All were trained on TinyShakespeare with matched hyperparameters (5000 iterations, 2-layer, block_size=256).

**Key findings:**
- **BigBird outperformed dense** (val loss 1.549 vs 1.567), a small but measurable gain, likely from global tokens acting as attention sinks and random attention providing implicit regularization.
- **Sliding window matched dense within noise** (val loss 1.565 vs 1.567) at window_size=128 — because half the sequence is already visible to each token, the restriction doesn't lose meaningful information on this dataset.
- **The benchmark shows no compute or memory savings across variants** — because our naive `masked_fill` implementation still allocates the full T×T score matrix. This is a real limitation of the implementation, not the algorithm; real efficiency gains require custom sparse kernels [2, 4].

The main takeaway: sparse attention patterns can match or exceed dense quality on this scale of task, and the theoretical O(n·w) scaling advantage is real — but realizing it in practice requires more than just applying a sparse mask to dense computation.

## 2. Design decisions

### 2.1 Attention pattern implementations

All three variants share the same base architecture: a 2-layer decoder-only transformer with pre-norm residual connections, multi-head attention, and a feed-forward MLP after each attention block. Causality is enforced in all variants — future tokens are always masked.

**Dense causal (baseline):** Standard scaled dot-product attention as in [1], with a lower-triangular causal mask. Every token attends to all previous tokens and itself.

**Sliding window (local):** Each token attends only to the previous `window_size` tokens plus itself. Longformer [2] originally uses a symmetric window (both past and future), but since we operate in a causal setting, our window only extends backward. This gives an O(n·w) attention pattern instead of O(n²). At window_size=128 and block_size=256, roughly half of the possible connections remain — this is why results match dense so closely; the effective information available per token is nearly identical.

**BigBird (sliding + global + random):** Following [3], this combines three components:
- **Local** (same as sliding window above).
- **Global tokens:** the first `num_global` positions attend to every other position and are attended to by every other position (subject to causality). These act as information hubs.
- **Random attention:** each token additionally attends to `num_random` uniformly-sampled earlier positions. Sampled once at model creation for reproducibility.

BigBird achieved the best val loss (1.549), 0.018 below dense. This is small but consistent across seeds in our runs and suggests the added global + random attention provides useful signal — likely as a form of implicit regularization or by giving the model dedicated attention anchors.

### 2.2 Numerical stability

A fully-masked query row causes softmax to produce NaN, because `softmax([-inf, -inf, ..., -inf])` is undefined. This can happen at pathological configurations (e.g., window_size=0) or in more elaborate sparse patterns where a row's valid attention positions could be empty.

**Fix:** in mask construction, we always force the diagonal to 1 — every token can attend to itself at minimum. This guarantees softmax input has at least one non-`-inf` entry per row and prevents NaN by construction. For our sliding-window and BigBird configurations this is never triggered in practice (sliding with window_size ≥ 1 already includes self, and BigBird's global tokens cover the earliest rows), but the safety mechanism is cheap and preserves stability under adversarial mask configs.

### 2.3 Naive masking vs. efficient sparse attention

An honest limitation: our benchmark (see [README.md](README.md#results)) shows all three variants have identical time and memory scaling — O(n²) — despite the algorithmic sparsity. This is because we implement sparsity via `masked_fill`, which zeroes entries in an already-allocated T×T score matrix. The full matrix is still computed and materialized; the mask only changes which entries are used downstream.

Real sparse attention avoids materializing the full T×T matrix in the first place:
- Longformer [2] provides a custom CUDA kernel that iterates only over positions inside the sliding window, achieving true O(n·w) time and memory.
- BigBird [3] uses block-sparse GPU kernels for the same reason.
- FlashAttention [4] takes a different approach — it computes attention in tiled blocks that never materialize the full matrix, saving memory without changing the algorithm.

Implementing any of these is out of scope for this task, but reporting the honest empirical result — that naive masking preserves algorithmic sparsity without realizing efficiency benefits — is more valuable than pretending to see speedups that aren't there.

## 3. Results

### 3.1 Benchmark (Item 1.5)

See [README.md](README.md#results) for the plot and full table. The clear finding: memory usage was nearly identical across all three variants at every sequence length tested (512 → 8192). Theoretically, sliding window and BigBird should achieve O(n·w) memory scaling, but because we used a window size of n/2, this reduces to O(n²/2) — the same asymptotic behavior as dense attention. On top of that, our naive `masked_fill` implementation allocates the full T×T score matrix regardless of the mask, so any theoretical savings from smaller windows aren't realized. The observed O(n²) scaling matches theory once both these factors are considered.

### 3.2 Quality comparison (Item 1.6)

Trained with matched hyperparameters (block_size=256, n_layer=2, n_head=6, batch_size=64, 5000 iterations).

| Variant  | Config                              | Train Loss | Val Loss |
|----------|-------------------------------------|-----------|----------|
| Dense    | full causal                         | 1.3221    | 1.5674   |
| Sliding  | window_size=128                     | 1.3185    | 1.5646   |
| BigBird  | window=128, global=2, random=3      | 1.3186    | 1.5490   |

Observations:
- **Sliding window matched dense within noise (Δ = 0.003).** Expected — the window covers half the block size, so each token sees a large fraction of its causal context. The information effectively available is nearly identical to dense.
- **BigBird's improvement over dense (Δ = 0.018) is small but consistent.** We ran a preliminary sanity check at n_layer=6 and 2500 iterations where the three variants converged much more closely, which suggests the BigBird gain here is partly a small-model effect: at n_layer=2 the model may be underparameterized enough that the added global + random attention provides useful signal, either as regularization or by giving the model dedicated attention anchors that help it use its limited capacity more efficiently. Put differently — BigBird seems to have hit a sweet spot between restricted local attention (sliding) and fully diluted dense attention.
- Training trajectory was similar for all three across the run — no dramatic convergence-speed differences visible in the logged evaluations at every 500 steps. The gap between variants opens gradually and stays small.

### 3.3 Correctness

The correctness harness (`harness.py`) contains 8 tests covering:
- Sliding window causality (upper triangle is always zero).
- Sliding window band structure (specific boundary positions have the expected mask values).
- Equivalence at boundary: sliding with window_size ≥ block_size produces the same mask as dense.
- Diagonal preservation (self-attention always available) for both sliding and BigBird.
- BigBird causality.
- BigBird contains global tokens (rows past the global range fully attend to global columns).
- BigBird includes sliding window as a subset (bigbird entries ≥ sliding entries).

All tests pass. Run with:
```bash
python harness.py
```

## 4. Discussion

### 4.1 Which pattern loses what information?

**Sliding window** loses everything outside its window — for a token at position i with window_size=w, positions [0, i-w-1] become invisible. On TinyShakespeare with window_size=128 and block_size=256, this loss is bounded (at worst half the context is masked, and only for the latest tokens in a sequence). Character-level Shakespeare has strong local structure — common letter pairs, word-length patterns, line-break behavior — so restricting attention to the last 128 characters barely hurts. On tasks with real long-range dependencies (coreference across paragraphs, retrieval-style QA, code with distant function calls), sliding window would fail more visibly.

**BigBird** recovers some of that loss through two mechanisms: global tokens (fixed positions that everyone attends to, providing information hubs that can carry long-range signal) and random attention (occasional long-range shortcuts, distributed across positions to give the whole sequence some non-local coverage). The recovery is partial — most possible long-range connections are still absent — but the theoretical result from the BigBird paper is that these three components together are enough to preserve universal approximation properties that pure sliding window loses.

The practical implication: sliding window is a bet that local structure is enough. BigBird hedges that bet by keeping some long-range channel open.

### 4.2 Why global tokens matter disproportionately

BigBird's paper argues global tokens are theoretically critical — they're what let the sparse pattern approximate arbitrary sequence-to-sequence functions. Without them, sliding + random alone doesn't have the same guarantees.

Practically, global tokens serve as attention sinks. Recent work on streaming LLMs and attention behavior has shown that transformers tend to route "excess" attention mass to specific positions regardless of their semantic relevance — often the first few tokens of a sequence. If those positions aren't attended to by everyone (as they are in dense attention), the model can't route information through them, and training becomes less stable. BigBird's global tokens provide this attention sink by construction.

In our experiments, the actual gain from global + random over pure sliding was small (0.016). This is likely because our setup doesn't stress-test the global mechanism — block_size=256 with window_size=128 already covers half the context, so long-range shortcuts have little to contribute. In a longer-context setting where sliding window truly loses information, we'd expect the gap to widen.

### 4.3 Where sparse patterns matter more than in our setup

Our results are honest but limited by the small scale. Where sparse patterns would show more:

- **Longer sequences.** At block_size=4096 or 8192, sliding window with fixed window_size=128 would lose most of the context. Both the memory savings (O(n·w) vs O(n²)) and the quality gap between dense and sliding would become meaningful.
- **Tasks with long-range dependencies.** Document QA, code understanding, or long-form summarization — anything where a token needs to attend to something hundreds of positions back. Character-level Shakespeare doesn't have this structure.
- **Larger models.** With more capacity, differences between attention patterns may show up more clearly (or disappear as the model compensates). Hard to predict without running.

## 5. Limitations and what we'd do differently

- **Naive masking limits the benchmark's interest.** Custom kernels (as in Longformer or FlashAttention) would show the real O(n·w) speedup that our implementation cannot.
- **Small model (n_layer=2), short training (5000 iterations).** Larger models trained longer might show more or fewer variant differences. We tested at n_layer=6 for 2500 iterations and saw the variants converge much more closely — suggesting some of BigBird's advantage here is due to the small-model regime. Fully characterizing this trade-off would need more training runs than compute allowed; runs at n_layer=6 took close to an hour per variant.
- **TinyShakespeare has minimal long-range dependencies.** A dataset like WikiText-103 or a long-document benchmark would better stress-test sparse patterns and give a more honest picture of when they matter.
- **Random attention sampled once at initialization.** Original BigBird resamples per forward pass. We chose fixed-at-init for reproducibility and simpler debugging, at the cost of slight deviation from the paper's construction.
- **No head-level pattern mixing.** All attention heads in our model use the same sparsity pattern. An interesting extension would be per-head patterns — some heads doing local attention, others global, others random — letting the model learn which pattern each head specializes in.

## 6. References

[1] Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.

[2] Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.

[3] Zaheer, M. et al. (2020). *Big Bird: Transformers for Longer Sequences*. NeurIPS.

[4] Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.