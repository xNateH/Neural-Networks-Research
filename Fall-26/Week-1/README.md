# Fall 2026 — Week 1: The Weight Matrix as an Embedding

**Nathan Haile** · Pattern Interpretability in Neural Networks ("Ideal State" project)

Prior weeks embedded layer **activations**. This week embeds the **parameters** themselves:
snapshot $\theta$ every epoch, and study the training trajectory, the pace of training, and where
different initializations start relative to converged solutions.

📄 **Start here:** [`NN_Research_Fall26_Week_1.pdf`](NN_Research_Fall26_Week_1.pdf) — 11-page write-up
with all tables and figures. If you only read one thing, read this.

---

## What was run

| | |
|---|---|
| **Runs** | 42 = 7 initializations × 2 depths × 3 seeds |
| **Architectures** | depth 26 (where PyTorch-default collapses) and depth 10 (healthy control), width 128 |
| **Initializations** | `pytorch_default`, `he_normal`, `he_uniform`, `xavier_normal`, `orthogonal`, `lognormal_he_scale`, `lognormal_jaynes` |
| **Data / optimizer** | MNIST, Adam (lr 1e-3), batch 256, 10 epochs, `weight_decay=0`, no scheduler |
| **Snapshots** | epoch 0 (pre-training) + every epoch = 11 per run, 462 total |
| **Extra** | 280 untrained networks (20 seeds × 7 schemes × 2 depths) for the "dead models, no training" case |

---

## The three questions, and the short answers

**Q1 — Can we see the trajectory and read the pace?**
Yes, but the obvious statistics are worthless. `Spearman(PC1, epoch) = 1.000` is reproduced by a
pure random walk in **100% of 200 simulations** (forced by *d* ≫ *n*) *and* by a network that never
learns. What survives is the consecutive-step cosine: **+0.030** for healthy training vs
**−0.146** for the dead run, which anti-aligns. Real finding: the network keeps moving at a
near-constant rate long after accuracy plateaus — epoch 1 takes 10.3% → 93.9%, the next nine
epochs travel 5× as far for +3.4 points.

**Q2 — Do initializations occupy distinguishable starting points?**
Yes, though largely by construction (these schemes differ in variance by design). Initialization
scale at epoch 0 predicts depth-26 survival at ρ = **+0.964**. Useful sub-finding:
`lognormal_he_scale` survives depth 26 at **97.0%** while `lognormal_jaynes` degrades to
**79.2%** — same distribution shape, different scale. **For log-normal init, scale matters more
than shape.**

**Q3 — Does any initialization start closer to the minima?**
No — but as a *bound*, not an equality. All seven schemes land in
`D_norm ∈ [0.9995, 1.0006]`, narrower than the measurement noise. A positive control shows the
metric only fires above ~50% true overlap. **Important caveat: the null preserves per-layer weight
marginals by construction, so this result is blind to distributional matching and does *not* test
the log-normal hypothesis.**

**Cleanest result (methodological):** SGD does **not** permute neurons mid-training — **0.0%** of
neurons move between consecutive epochs of one run, vs **91.4%** across seeds. Within-run
trajectories can be read directly; any cross-run weight comparison requires alignment first.

---

## How to run it

Everything is in one self-contained notebook:
**[`week1_weight_space_trajectory.ipynb`](week1_weight_space_trajectory.ipynb)**

```bash
pip install torch torchvision numpy pandas scipy matplotlib nbformat jupyter
jupyter notebook week1_weight_space_trajectory.ipynb   # then Run All
```

MNIST downloads automatically to `./data/` on first run.

**Runtime:** ~25 min on CPU for the full sweep (42 runs), then ~3 min of analysis.
The notebook is **resume-safe** — it writes weight snapshots to `snapshots/` and skips any run
already on disk, so re-running only redoes the analysis.

> **Note:** the ~1 GB of raw weight snapshots is *not* in this repo (742 files, far past GitHub's
> limits). They regenerate deterministically from fixed seeds. All *derived* results — every CSV,
> figure, and the report — are committed, so you can inspect every number without running anything.

CPU is intentional: at 128×128 matmuls it benchmarked ~3× faster than MPS, since kernel-launch
overhead dominates at this size.

---

## Notebook structure

| Section | Contents |
|---|---|
| 1 | Architecture + the 7 initialization schemes |
| 2 | **Permutation alignment** (Git Re-Basin weight matching) + 4 unit tests |
| 3 | Training protocol and snapshotting |
| 4 | Gram-matrix PCA embeddings (absolute + displacement) |
| 5 | Distance analysis and the **null baseline** |
| 6 | **Controls** — random-walk null, dead-run control, Q3 positive control |
| 7 | Untrained population (20 seeds) |
| — | **RESULTS**: 10 figures, 7 tables, 8 validation checks |
| A | Appendix: why `weight_decay = 0` |

### Why permutation alignment is necessary

Hidden neurons can be permuted without changing the function a network computes, so two networks
that learned the same thing can sit far apart in raw parameter space purely because their neurons
are indexed differently. For permutation matrices $P_1..P_{L-1}$ (one per hidden layer, with the
784-dim input and 10-dim output held fixed):

$$W_l \rightarrow P_l W_l P_{l-1}^\top, \qquad b_l \rightarrow P_l b_l$$

Alignment solves for these via layer-wise Hungarian matching (Ainsworth et al., 2022). Unit test
**U1** verifies a permutation leaves network output unchanged (max deviation 1.9e-6).

### Why every headline number has a null

Aligning two networks *searches* over permutations and shrinks distance **for free** — 15.5%
between two independent random networks, and more for heavy-tailed initializations, which is
exactly what log-normal schemes are. Ranked by *raw* aligned distance the apparent winner is
`pytorch_default` — **the dead scheme** — purely because its weights are smallest (ρ = −0.893 with
weight magnitude). Three results this week looked publishable and did not survive their controls;
all three are documented in §7.1 of the report.

---

## Files

| File | What it is |
|---|---|
| `week1_weight_space_trajectory.ipynb` | **The experiment** — one self-contained notebook, executed with outputs |
| `NN_Research_Fall26_Week_1.pdf` / `.tex` | Weekly report |
| `build_notebook.py` | Generator that emits the notebook (source of truth for the code) |
| `MANIFEST.csv` | Index of every figure → its point-data CSV |
| `fig*.png` + `fig*_points.csv` | Each figure, and the **exact plotted values** behind it |
| `snapshots.csv` | Per-snapshot metrics: PCA coords, norms, lens metrics, weight stats |
| `pairwise.csv` | Init → trained-reference distances (aligned, unaligned, null-normalized) |
| `minima_cloud.csv` | Pairwise distances between converged solutions |
| `drift.csv` | Permutation drift within vs. across runs |
| `untrained_population.csv` | 20-seed untrained population statistics |
| `validation_checks.csv` | 8 assertions confirming the setup reproduces known behaviour |
| `appendixA_wd_probe.csv` | Weight-decay probe justifying `wd=0` |
| `histories/` | Per-run accuracy and loss curves (JSON) |

Every figure has a matching `_points.csv` so any plotted value can be checked directly.

---

## Validation

8 checks, all passing — run at the end of the notebook:

| Check | Result |
|---|---|
| V1 `pytorch_default` collapses at depth 26 | 11.35% ✓ |
| V2 He/Xavier survive at depth 26 | 96.97% / 96.77% ✓ |
| V3 depth-10 control: all schemes train | min 96.85% ✓ |
| V4 healthy runs travel ≥3× further than dead | 7.1× ✓ |
| V5 dead runs stay at init scale (`wd=0`) | 0.999× ✓ |
| V6 Mirsky bound `d_aligned ≥ D_spec` | 0 violations / 42 ✓ |
| V7 no permutation drift within a run | 0.0000 moved ✓ |
| V8 alignment does matter across runs | 10.99% reduction ✓ |

---

## Next step

The non-trivial version of Q2 — *does seed-level position at epoch 0 predict survival within a
single scheme?* — needs a configuration on the collapse boundary where some seeds live and some
die under an identical scheme. Week 16's depth-30 / width-512 + BatchNorm setup (which oscillated
between 55% and 96%) is the natural candidate. Separately, a null that is **not** blind to
marginal distribution would let Q3 finally speak to the log-normal hypothesis directly.

---

### References

- Ainsworth, Hayase & Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries.* arXiv:2209.04836, 2022.
- Li, Xu, Taylor, Studer & Goldstein. *Visualizing the Loss Landscape of Neural Nets.* NeurIPS, 2018.
- Venkatasubramanian, Sanjeevrajan & Khandekar. *Jaynes Machine: The Universal Microstructure of Deep Neural Networks.* arXiv:2310.06960, 2023.
