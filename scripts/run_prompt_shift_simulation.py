"""Synthetic prompt-shift benchmark for PromptShift-CRC.

The benchmark creates prompt embeddings with recurring user/task regimes and
abrupt domain shifts. Risk is higher when the current prompt comes from a
harder or shifted regime. The script compares static CRC, rolling CRC,
adaptive CRC, representation-weighted CRC, and PromptShift-CRC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / np.sum(weights)
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, q, side="left")])


def make_stream(seed: int = 7, n: int = 2400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = 10
    embeddings = np.zeros((n, dim))
    risk = np.zeros(n)
    regimes = np.zeros(n, dtype=int)

    centers = np.array(
        [
            np.r_[0.0, 0.0, 0.0, np.zeros(dim - 3)],
            np.r_[1.2, 0.7, 0.0, np.zeros(dim - 3)],
            np.r_[0.0, -1.1, 1.0, np.zeros(dim - 3)],
            np.r_[-1.0, 0.2, -1.2, np.zeros(dim - 3)],
        ]
    )
    difficulties = np.array([0.18, 0.32, 0.48, 0.62])

    for t in range(n):
        if t < 600:
            regime = (t // 75) % 2
        elif t < 1200:
            regime = 2
        elif t < 1800:
            regime = 1 if (t // 60) % 2 == 0 else 3
        else:
            regime = 0 if (t // 50) % 3 == 0 else 3

        regimes[t] = regime
        center = centers[regime]
        local_shift = 0.35 * np.sin(2 * np.pi * t / 180.0)
        embeddings[t] = center + local_shift + rng.normal(scale=0.65, size=dim)

        novelty = min(1.0, np.linalg.norm(embeddings[t] - centers[0]) / 4.0)
        burst = 0.18 if 1150 <= t < 1300 or 1780 <= t < 1920 else 0.0
        base = difficulties[regime] + 0.18 * novelty + burst
        risk[t] = np.clip(base + rng.normal(scale=0.09), 0.0, 1.0)

    return embeddings, risk, regimes


def evaluate(seed: int = 7, n: int = 2400, alpha: float = 0.1) -> tuple[list[dict[str, float | str]], dict[str, np.ndarray]]:
    embeddings, risk, regimes = make_stream(seed=seed, n=n)

    rows: list[dict[str, float | str]] = []
    traces: dict[str, np.ndarray] = {"risk": risk, "regime": regimes}
    cal_start = 300
    window = 300
    recent_window = 160
    bandwidth = 1.8
    target = 1.0 - alpha

    for method in ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]:
        misses = []
        thresholds = []
        drift_scores = []
        ess_values = []
        alpha_t = alpha

        for t in range(cal_start, n):
            if method == "static":
                idx = np.arange(0, cal_start)
                weights = np.ones(len(idx)) / len(idx)
                drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
            elif method == "static_crc":
                idx = np.arange(0, cal_start)
                weights = np.ones(len(idx)) / len(idx)
                drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
            elif method in {"rolling_crc", "adaptive_crc"}:
                idx = np.arange(max(0, t - window), t)
                weights = np.ones(len(idx)) / len(idx)
                drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
            elif method == "representation_weighted_crc":
                idx = np.arange(max(0, t - 3 * window), t)
                dist = np.linalg.norm(embeddings[idx] - embeddings[t], axis=1)
                raw = np.exp(-(dist**2) / bandwidth**2)
                weights = raw + 1e-8
                weights = weights / weights.sum()
                drift = float(np.average(dist, weights=weights))
            else:
                idx = np.arange(max(0, t - 4 * window), t)
                dist = np.linalg.norm(embeddings[idx] - embeddings[t], axis=1)
                sim = np.exp(-(dist**2) / bandwidth**2)
                recent_idx = np.arange(max(0, t - recent_window), t)
                recent_mask = np.isin(idx, recent_idx).astype(float)
                recent = recent_mask / max(1.0, recent_mask.sum())
                sim = sim / max(sim.sum(), 1e-12)
                local_drift = float(np.average(dist, weights=sim + 1e-12))
                gate = 1.0 / (1.0 + np.exp(2.0 * (local_drift - 2.0)))
                weights = gate * sim + (1.0 - gate) * recent
                weights = weights + 1e-8
                weights = weights / weights.sum()
                drift = local_drift

            current_target = 1.0 - alpha_t
            threshold = weighted_quantile(risk[idx], weights, current_target)
            miss = float(risk[t] > threshold)
            if method in {"adaptive_crc", "promptshift_crc"}:
                alpha_t = float(np.clip(alpha_t + 0.025 * (alpha - miss), 0.02, 0.35))
            ess = float(1.0 / np.sum(weights**2))
            misses.append(miss)
            thresholds.append(threshold)
            drift_scores.append(drift)
            ess_values.append(ess)

        rows.append(
            {
                "method": method,
                "coverage": 1.0 - float(np.mean(misses)),
                "mean_threshold": float(np.mean(thresholds)),
                "mean_drift": float(np.mean(drift_scores)),
                "mean_effective_calibration_size": float(np.mean(ess_values)),
            }
        )
        traces[f"{method}_miss"] = np.array(misses)
        traces[f"{method}_threshold"] = np.array(thresholds)
        traces[f"{method}_drift"] = np.array(drift_scores)
        traces[f"{method}_ess"] = np.array(ess_values)

    return rows, traces


def write_summary(rows: list[dict[str, float | str]]) -> None:
    out = RESULTS / "prompt_shift_pilot_summary.csv"
    with out.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def make_figure(traces: dict[str, np.ndarray], cal_start: int = 300) -> None:
    if plt is None:
        make_figure_pil(traces, cal_start=cal_start)
        return
    t = np.arange(cal_start, len(traces["risk"]))
    methods = ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]
    labels = ["Static CRC", "Rolling CRC", "Adaptive CRC", "Rep-weighted CRC", "PromptShift-CRC"]
    colors = ["#7a7a7a", "#2a6fbb", "#1b9e77", "#984ea3", "#d95f02"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(np.arange(len(traces["risk"])), traces["risk"], color="#222222", lw=0.8, alpha=0.7)
    for x in [600, 1200, 1800]:
        axes[0].axvline(x, color="#b22222", ls="--", lw=1)
    axes[0].set_ylabel("Risk score")
    axes[0].set_title("Synthetic prompt/domain shift benchmark")

    for method, label, color in zip(methods, labels, colors):
        miss = traces[f"{method}_miss"]
        local_miss = np.convolve(miss, np.ones(120) / 120, mode="same")
        axes[1].plot(t, local_miss, label=label, color=color, lw=1.4)
    axes[1].axhline(0.1, color="black", ls=":", lw=1)
    axes[1].set_ylabel("Local violation rate")
    axes[1].legend(ncol=2, fontsize=8)

    axes[2].plot(t, traces["promptshift_crc_drift"], color="#d95f02", lw=1.2, label="PromptShift drift")
    axes[2].set_ylabel("Drift score")
    axes[2].set_xlabel("Time")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "figure1_promptshift_crc_pilot.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def _scale(values: np.ndarray, lo: float, hi: float, y0: int, y1: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return y1 - (values - lo) / max(hi - lo, 1e-9) * (y1 - y0)


def make_figure_pil(traces: dict[str, np.ndarray], cal_start: int = 300) -> None:
    if Image is None:
        return
    width, height = 1400, 950
    margin_left, margin_right = 95, 45
    panels = [(90, 300), (370, 630), (710, 880)]
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    title_font = font
    methods = ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]
    labels = ["Static CRC", "Rolling CRC", "Adaptive CRC", "Rep-weighted CRC", "PromptShift-CRC"]
    colors = ["#777777", "#2a6fbb", "#1b9e77", "#984ea3", "#d95f02"]

    def xcoord(t: np.ndarray) -> np.ndarray:
        return margin_left + (t / (len(traces["risk"]) - 1)) * (width - margin_left - margin_right)

    draw.text((margin_left, 28), "PromptShift-CRC synthetic prompt/domain shift benchmark", fill="#111111", font=title_font)

    # Panel 1: risk score.
    y0, y1 = panels[0]
    draw.rectangle((margin_left, y0, width - margin_right, y1), outline="#d0d0d0")
    risk = traces["risk"]
    xs = xcoord(np.arange(len(risk)))
    ys = _scale(risk, 0, 1, y0 + 15, y1 - 15)
    pts = list(zip(xs.astype(int), ys.astype(int)))
    draw.line(pts, fill="#222222", width=2)
    for shift in [600, 1200, 1800]:
        x = int(xcoord(np.array([shift]))[0])
        draw.line((x, y0, x, y1), fill="#b22222", width=2)
    draw.text((20, (y0 + y1) // 2), "Risk", fill="#111111", font=font)

    # Panel 2: local violation rates.
    y0, y1 = panels[1]
    draw.rectangle((margin_left, y0, width - margin_right, y1), outline="#d0d0d0")
    t = np.arange(cal_start, len(risk))
    xs = xcoord(t)
    for method, color in zip(methods, colors):
        miss = traces[f"{method}_miss"]
        local = np.convolve(miss, np.ones(120) / 120, mode="same")
        ys = _scale(local, 0, 0.75, y0 + 15, y1 - 15)
        draw.line(list(zip(xs.astype(int), ys.astype(int))), fill=color, width=2)
    y_target = int(_scale(np.array([0.1]), 0, 0.75, y0 + 15, y1 - 15)[0])
    draw.line((margin_left, y_target, width - margin_right, y_target), fill="#111111", width=1)
    draw.text((20, (y0 + y1) // 2), "Local violation", fill="#111111", font=font)
    lx, ly = margin_left + 15, y0 + 12
    for label, color in zip(labels, colors):
        draw.line((lx, ly + 5, lx + 30, ly + 5), fill=color, width=3)
        draw.text((lx + 38, ly), label, fill="#111111", font=font)
        ly += 22

    # Panel 3: drift and effective size.
    y0, y1 = panels[2]
    draw.rectangle((margin_left, y0, width - margin_right, y1), outline="#d0d0d0")
    drift = traces["promptshift_crc_drift"]
    ys = _scale(drift, float(np.min(drift)), float(np.max(drift)), y0 + 15, y1 - 15)
    draw.line(list(zip(xs.astype(int), ys.astype(int))), fill="#d95f02", width=2)
    draw.text((20, (y0 + y1) // 2), "Drift", fill="#111111", font=font)
    draw.text((margin_left, height - 42), "Time", fill="#111111", font=font)

    out = FIGURES / "figure1_promptshift_crc_pilot.png"
    img.save(out)
    print(f"Wrote {out}")


def main() -> None:
    rows, traces = evaluate()
    write_summary(rows)
    make_figure(traces)


if __name__ == "__main__":
    main()
