"""Ablation study for PromptShift-CRC.

This script quantifies the contribution of the drift gate, recent-window
fallback, representation weighting, and online risk update.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

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


def make_stream(seed: int = 31, n: int = 2400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = 10
    centers = np.array(
        [
            np.r_[0.0, 0.0, 0.0, np.zeros(dim - 3)],
            np.r_[1.3, 0.4, -0.2, np.zeros(dim - 3)],
            np.r_[-0.3, -1.3, 1.1, np.zeros(dim - 3)],
            np.r_[1.0, -0.9, -1.0, np.zeros(dim - 3)],
        ]
    )
    difficulty = np.array([0.16, 0.30, 0.51, 0.64])
    embeddings = np.zeros((n, dim))
    risk = np.zeros(n)
    for t in range(n):
        if t < 600:
            regime = 0 if (t // 80) % 2 == 0 else 1
        elif t < 1200:
            regime = 2
        elif t < 1800:
            regime = 1 if (t // 75) % 2 == 0 else 3
        else:
            regime = 0 if (t // 70) % 3 == 0 else 3
        seasonal = 0.25 * np.sin(2 * np.pi * t / 220)
        embeddings[t] = centers[regime] + seasonal + rng.normal(scale=0.62, size=dim)
        novelty = min(1.0, np.linalg.norm(embeddings[t] - centers[0]) / 4.5)
        burst = 0.16 if 1130 <= t < 1270 or 1760 <= t < 1910 else 0.0
        risk[t] = np.clip(difficulty[regime] + 0.16 * novelty + burst + rng.normal(scale=0.085), 0.0, 1.0)
    return embeddings, risk


def evaluate_variant(
    embeddings: np.ndarray,
    risk: np.ndarray,
    variant: str,
    alpha: float = 0.1,
) -> dict[str, float | str]:
    n = len(risk)
    cal_start = 300
    window = 300
    recent_window = 160
    bandwidth = 1.85
    alpha_t = alpha
    misses: list[float] = []
    thresholds: list[float] = []
    ess_values: list[float] = []
    drift_values: list[float] = []

    for t in range(cal_start, n):
        if variant == "rolling_only":
            idx = np.arange(max(0, t - window), t)
            weights = np.ones(len(idx)) / len(idx)
            drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
        else:
            idx = np.arange(max(0, t - 4 * window), t)
            dist = np.linalg.norm(embeddings[idx] - embeddings[t], axis=1)
            if variant == "no_representation_weighting":
                sim = np.ones(len(idx)) / len(idx)
            else:
                sim = np.exp(-(dist**2) / bandwidth**2)
                sim /= max(sim.sum(), 1e-12)

            recent_idx = np.arange(max(0, t - recent_window), t)
            recent_mask = np.isin(idx, recent_idx).astype(float)
            recent = recent_mask / max(1.0, recent_mask.sum())
            local_drift = float(np.average(dist, weights=sim + 1e-12))

            if variant == "no_drift_gate":
                gate = 0.5
            else:
                gate = 1.0 / (1.0 + np.exp(2.2 * (local_drift - 2.05)))

            if variant == "no_recent_fallback":
                weights = sim + 1e-8
            else:
                weights = gate * sim + (1.0 - gate) * recent + 1e-8
            weights /= weights.sum()
            drift = local_drift

        q = weighted_quantile(risk[idx], weights, 1.0 - alpha_t)
        miss = float(risk[t] > q)
        if variant != "no_online_update" and variant != "rolling_only":
            alpha_t = float(np.clip(alpha_t + 0.02 * (alpha - miss), 0.02, 0.35))
        misses.append(miss)
        thresholds.append(q)
        ess_values.append(float(1.0 / np.sum(weights**2)))
        drift_values.append(drift)

    return {
        "variant": variant,
        "coverage": 1.0 - float(np.mean(misses)),
        "mean_threshold": float(np.mean(thresholds)),
        "mean_effective_calibration_size": float(np.mean(ess_values)),
        "mean_drift": float(np.mean(drift_values)),
        "absolute_coverage_error": abs((1.0 - float(np.mean(misses))) - 0.9),
    }


def run() -> list[dict[str, float | str]]:
    embeddings, risk = make_stream()
    variants = [
        "rolling_only",
        "no_representation_weighting",
        "no_drift_gate",
        "no_recent_fallback",
        "no_online_update",
        "full_promptshift_crc",
    ]
    return [evaluate_variant(embeddings, risk, v) for v in variants]


def write_summary(rows: list[dict[str, float | str]]) -> None:
    out = RESULTS / "ablation_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def make_figure(rows: list[dict[str, float | str]]) -> None:
    if Image is None:
        return
    width, height = 1200, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((70, 35), "PromptShift-CRC ablation: coverage error by removed component", fill="#111111", font=font)

    labels = {
        "rolling_only": "Rolling only",
        "no_representation_weighting": "No representation",
        "no_drift_gate": "No drift gate",
        "no_recent_fallback": "No recent fallback",
        "no_online_update": "No online update",
        "full_promptshift_crc": "Full PromptShift",
    }
    colors = {
        "rolling_only": "#777777",
        "no_representation_weighting": "#984ea3",
        "no_drift_gate": "#2a6fbb",
        "no_recent_fallback": "#1b9e77",
        "no_online_update": "#e6ab02",
        "full_promptshift_crc": "#d95f02",
    }
    plot_left, plot_top, plot_right, plot_bottom = 90, 90, 1120, 560
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#d0d0d0")
    max_err = max(float(r["absolute_coverage_error"]) for r in rows) * 1.2
    bar_w = (plot_right - plot_left) / (len(rows) * 1.45)
    for i, row in enumerate(rows):
        err = float(row["absolute_coverage_error"])
        x0 = int(plot_left + i * bar_w * 1.45 + 20)
        x1 = int(x0 + bar_w)
        y = int(plot_bottom - err / max_err * (plot_bottom - plot_top))
        draw.rectangle((x0, y, x1, plot_bottom), fill=colors[str(row["variant"])])
        draw.text((x0 - 8, plot_bottom + 20), labels[str(row["variant"])], fill="#111111", font=font)
        draw.text((x0, y - 18), f"{err:.3f}", fill="#111111", font=font)
    draw.text((35, 280), "|coverage - 0.90|", fill="#111111", font=font)
    out = FIGURES / "figure3_ablation_summary.png"
    img.save(out)
    print(f"Wrote {out}")


def main() -> None:
    rows = run()
    write_summary(rows)
    make_figure(rows)


if __name__ == "__main__":
    main()
