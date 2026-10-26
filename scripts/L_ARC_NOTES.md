# L-ARC baseline: notes for `baselines_larc.py`

Method: Zecchin and Simeone, "Localized Adaptive Risk Control", arXiv:2405.07976 (v3 read). Authors' code: github.com/kclip/localized-adaptive-risk-control (not cloned, downloaded or run).

## How the sources were read (read this first)

Both the paper (arxiv.org/html/2405.07976v3) and the code (raw GitHub files) were read through WebFetch, which returns a small model's answer to a question, not the file. I asked for raw LaTeX / literal code lines where it mattered. The kernel line and the step-size line came back identically from separate fetches of the electricity, tumor, beam and fruit scripts. I did not see the files byte for byte. The arXiv PDF came back as binary and was not read. The last ~8.8k characters of the HTML (end of Appendix C) were read; nothing beyond that exists.

## Parameterisation as implemented

Harness orientation: `miss_t = 1{risk_t > g_t(x_t)}`, target violation rate `alpha`. The loss is non-increasing in the threshold, which is the paper's Assumption 3 orientation (larger threshold, larger set, smaller loss). The risk plays the role of the non-conformity score.

- `g_t(x) = f_t(x) + c_t`, `f_t(x) = sum_{s<t} beta_s * kappa * exp(-||x - x_s||^2 / ell^2)`, sum over post-prefix points only.
- `eta_t = eta1 / sqrt(t)`, t = 1 at the first post-prefix step (`t_offset` shifts it; default 0).
- `c_{t+1} = c_t - eta_t (alpha - miss_t)` (paper Eq. 17).
- `f_{t+1} = (1 - lam eta_t) f_t - eta_t (alpha - miss_t) k(x_t, .)` (paper Eq. 18): all old `beta_s` are multiplied by `1 - lam eta_t`, then `beta_t = -eta_t (alpha - miss_t)`. `kappa` multiplies the whole sum (equivalent to putting it in the kernel).
- Initialisation: `f_1 = 0`; `c_1 = np.quantile(risk[:cal_start], 1 - alpha)`, the same warm start as `pid_quantile_tracking`. The paper and the authors' code use `c_1 = 0` (`warm_start=False` gives that).
- Clipping: none in the update or in the miss indicator (the paper and the code do not clip). Only the reported `threshold` field is clipped to [0, 1] by the harness's `_store`; `raw_q` is unclipped.
- Cost: one kernel row (one matrix-vector product, d^2 from precomputed squared norms) and one O(t) shrink per step; coefficients are a preallocated array.
- Trace fields: `miss, threshold, raw_q` as above; `ess = float(cal_start)`, `drift = drift_pre = 0`, `alpha = alpha`, `inf = 0`, the same neutral values as `pid_quantile_tracking`. `lam = 1e-4`, `kappa = 1` are the paper's stated defaults ("Unless stated otherwise ... kappa = 1, length scale l = 1, lambda = 1e-4", Section 3).

### What `ell` means (the one real ambiguity)

- The authors' code (electricity, tumor, beam, fruit scripts) computes `exp(-||x - x'||^2 / length_scale**2)`.
- The paper writes `kappa exp(-||x - x'||^2 / l)` with l unsquared (Section 3, 3.1, 3.2, C.2; raised-RBF form with +1 in Section 2.3 and Eq. 10), and calls l the "length scale". It also gives `l = 2e(kappa/rho)^2` in Section 2.3, which is dimensionally a squared length.
- `baselines_larc.py` uses the code's form by default (`kernel_form="code"`: `ell` enters as `ell^2`). `kernel_form="paper"` divides by `ell` instead (then `l = ell^2`).
- At `ell = 1` the two forms are the same kernel, so "authors' default" is `exp(-||x - x'||^2)` under either reading.
- `matched_length_scale(h)` returns `h` (code form) or `h^2` (paper form), so the kernel equals `exp(-||x - x'||^2 / h^2)`, the kernel PromptShift uses with `h = cfg["bw"]`. Both forms give the same kernel for the matched configurations. The choice only changes the printed value of `ell`, not what runs, for all four configurations.
- The authors' ell values are relative to scaled features: MinMaxScaler on 7-day lag vectors (electricity); StandardScaler, PCA(5), then MinMaxScaler (tumor); scalar confidence in [0,1] (fruit). The harness passes unit-norm 384-dim embeddings, where two random unit vectors have `d^2 ~ 2` (measured mean 1.999). So `ell = 1` is a weakly local kernel here. This is my inference from the feature scaling; I did not measure the real MiniLM streams.

## The four configurations (`make_larc_configs()`, also `LARC_BASELINES` and `register(registry)`)

| name | eta1 | ell |
|---|---|---|
| `larc_eta0.1_ellDefault` | 0.1 | 1 |
| `larc_eta0.1_ellMatched` | 0.1 | `matched_length_scale(cfg["bw"])`, read from cfg at call time |
| `larc_eta1_ellDefault` | 1 | 1 |
| `larc_eta1_ellMatched` | 1 | as above |

All: `lam = 1e-4`, `kappa = 1`, `warm_start=True`. `eta1 = 1` is what every authors' script passes (`eta_init = 1.`). `eta1 = 0.1` is the paper's text for tumor segmentation. They are `functools.partial` objects, so they pickle (checked).

## Authors' code versus the paper (as read through the fetcher)

1. Kernel denominator: code `length_scale**2`, paper `l` (above). The beam-selection paper text says l = 10, and the code uses `length_scale = 10.`, which is a denominator of 100 under the code form and 10 under the paper's literal formula. I cannot tell which produced the paper's figure.
2. Tumor segmentation step size: paper text `eta_t = 0.1 t^{-1/2}`; `tumor_segmentation/main.py` sets `eta_init = 1.` and `decay = 1/2` and passes it to both ARC and L-ARC (two separate fetches agree). Electricity matches (`eta_t = t^{-1/2}` in the paper, `eta_init = 1.`).
3. Sign convention: electricity uses the harness orientation (`err = score > q`, `c <- c - eta (alpha - err)`). Tumor, beam and fruit define sets as `{Y_HAT > q}`, so the loss rises with q and the code uses `c <- c + eta (alpha - loss)`. This is a reparametrisation of the threshold direction, not a disagreement with the paper.
4. Iterate used at test time: the code returns last-iterate and time-averaged coefficients/intercept (`coeff_mean / N`, `mean(c)`) and the test thresholds in tumor and fruit use the averages (the fruit script has a comment pointing to the last iterate as the alternative; electricity's test loop was reported as using the last two returned values, which are the averages). The paper's Theorem 1 is about the averaged iterate; Theorem 2 about the online iterates. The harness evaluates online iterates.
5. `kappa` placement: inside the kernel (tumor, beam) versus multiplying the sum (electricity, fruit); equivalent.
6. Memory: the code has a `Mem_max` truncation (paper Appendix C.3 variant), default `2**32-1`, never reached; not implemented here (full memory).
7. Matches: `f_1 = 0, c_1 = 0`, `eta_t = eta_init / (t+1)**0.5` with 0-based t (same as `eta_1 t^{-1/2}` 1-based), `(1 - eta lam)` shrink of old coefficients before appending, no regularisation of the intercept, no clipping or projection, `lam = reg_param = 1e-4`, `kappa = 1`.

## Verified citations (from the fetched paper text)

- Eq. 17, Eq. 18: the updates above; `g_t = f_t + c_t`; threshold-class kernel `k' = k + 1`.
- Assumption 1: stationary kernel `k(x,x') = k~(||x-x'||)`, `k~ >= 0` rho-Lipschitz, bounded by kappa, decaying to 0. Assumption 2: scores in `[0, S_max]`. Assumption 3: loss in `[0, B]`, non-increasing in set size. Assumptions 4 and 5 (strictly decreasing, left-continuous) are for Theorem 1. The Gaussian kernel meets Assumption 1 (my observation).
- Theorem 1 (i.i.d. data, time-averaged threshold): `limsup E[ w(X)/E[w(X)] L(C(X, gbar_T), Y) ] <= alpha + kappa B ||f_w||_H / E[w(X)]` in probability, for `w = f_w + c_w` in the weighting class; `eta_t = eta_1 t^{-1/2} < 1/lambda`.
- Theorem 2 (Eq. 23; arbitrary sequence with `||X_t|| <= D`; online iterates): `|(1/T) sum_t L_t - alpha| <= T^{-1/2} ( S_max/eta_0 + 4B sqrt(rho kappa D)/(eta_0 lambda) + 2B(2 kappa + 1) ) + kappa B`. The statement writes the step size with `eta_1` but the bound with `eta_0`; the paper does not reconcile them (the proof uses `eta_t = eta_0 t^{-omega}`). Hypothesis `eta_t < 1/lambda`; with `lam = 1e-4` this holds for eta1 = 0.1 or 1.
- The `kappa B` term does not shrink with T; it is zero only for `kappa = 0` (recovering the ARC guarantee). With the default `kappa = 1` and the 0/1 violation loss (`B = 1`) the right side is at least 1, so Theorem 2 gives no usable bound on the violation rate for the default configurations (my arithmetic). The paper says kappa trades localisation against convergence to the long-term target and gives no rule for choosing it.
- Proposition 3: `g_t(x) in [G_min, G_max]` with `G_max = S_max + 2B sqrt(rho kappa D)/lambda + eta_0 B (2 kappa + 1)`, `G_min = -2B sqrt(rho kappa D)/lambda - eta_0 B (2 kappa + 1)`.
- Paper experiment settings (text): kappa = 1, l = 1, lambda = 1e-4 default; electricity T = 45312, alpha = 0.1, `eta_t = t^{-1/2}`; tumor T = 2098, alpha (FNR) = 0.1, `eta_t = 0.1 t^{-1/2}`; beam T = 25000, l = 10; image classification (C.2) alpha = 0.25, T = 8000.

## Tests run (`python -B baselines_larc.py`, one BLAS thread)

- Vectorised code equals a slow reference (explicit coefficient products, Python loops) to 5.6e-16; matched `ell` equals the explicit value in both forms.
- iid random unit vectors (384-d, 3000 steps, uniform risk): violation rate 0.1024 (eta1 = 0.1) and 0.0999 (eta1 = 1), target 0.1.
- Two interleaved regimes (orthogonal centres, risk U(0,0.4) versus U(0.5,1)): `ell = 0.5` gives violation 0.099 / 0.099 and mean thresholds 0.363 / 0.940; `ell = 100` gives 0.000 / 0.199 and 0.892 / 0.893 (no localisation). Matched `ell` at `bw = 1.2` also gives 0.099 / 0.099 on this easy geometry.
- Trace keys and shapes equal `pid_quantile_tracking`'s; `crc_lib.summarise` runs; 4000 online steps in 384-d took 6.2 s and 7.9 s on two runs.

## Not verified

- The source files themselves (summaries only); the PDF; the proofs.
- Whether the venue is NeurIPS 2024 (taken from the task, not checked).
- Paper text for beam-selection alpha and eta, and for the C.2 learning rate, kappa, lambda and l values (not found in the text read); the code values are in the code notes above. The paper lists no specific electricity length-scale values; the code runs ell in {2, 1, 0.5} (electricity, tumor), {1, 0.5, 0.25} (fruit), {10} (beam). Fruit and beam code were read only for kernel, step size, signs and settings.
- Agreement with the authors' implementation numerically: not tested (their code was not run); my check is against my own slow reference of Eq. 17-18.
- How the paper's guarantees translate to the warm-started, harness-evaluated stream: not analysed.

## Harness notes

- `baselines_online.py` (not edited) still says in its docstring that L-ARC is "not reimplemented". It is now outdated.
- To use the baselines in `run_real_streams.py`: after `import baselines_online as B` and before `METHODS` is built, add `import baselines_larc as L; L.register(B.BASELINES)`. `METHODS` already ends with `list(B.BASELINES)` and `job()` already dispatches on `m in B.BASELINES`, and registering at module level also happens in spawned Pool workers (my reading of the code, not run). `gc_bound` will be NaN for these names because `C.is_online` does not list them, as for the other baselines.
- Step size decays like `1/sqrt(t)` from the first post-prefix step, whereas the other online baselines use a constant `gamma`; L-ARC adapts more slowly late in long streams. `t_offset = cal_start` gives the step size a stream-position clock instead.
