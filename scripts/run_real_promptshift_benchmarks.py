"""LLM-style prompt-shift benchmark prototypes for PromptShift-CRC.

The script creates three structured benchmark streams that mimic common
foundation-model deployment settings:

1. question answering under topic shift,
2. summarization factuality under source-domain shift,
3. safety/toxicity screening under user-intent shift.

Each stream has prompt embeddings, domain labels, and risk scores. The setup is
designed so it can later be swapped with public LLM benchmark outputs while
keeping the calibration code unchanged.
"""

from __future__ import annotations

import csv
import zlib
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


def make_task_stream(task: str, seed: int = 11, n: int = 2200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + zlib.crc32(task.encode("utf-8")) % 997)
    dim = 12

    if task == "qa_topic_shift":
        centers = np.array([
            np.r_[0.0, 0.0, 0.0, np.zeros(dim - 3)],
            np.r_[1.4, 0.2, 0.4, np.zeros(dim - 3)],
            np.r_[0.2, -1.5, 1.1, np.zeros(dim - 3)],
            np.r_[-1.0, 1.0, -0.8, np.zeros(dim - 3)],
        ])
        difficulties = np.array([0.15, 0.26, 0.49, 0.58])
    elif task == "summarization_domain_shift":
        centers = np.array([
            np.r_[0.0, 0.0, 0.0, np.zeros(dim - 3)],
            np.r_[0.8, -1.0, 0.3, np.zeros(dim - 3)],
            np.r_[1.6, 0.7, -0.4, np.zeros(dim - 3)],
            np.r_[-1.2, -0.6, 1.3, np.zeros(dim - 3)],
        ])
        difficulties = np.array([0.18, 0.31, 0.52, 0.66])
    else:
        centers = np.array([
            np.r_[0.0, 0.0, 0.0, np.zeros(dim - 3)],
            np.r_[-0.6, 1.2, 0.5, np.zeros(dim - 3)],
            np.r_[1.0, 1.1, -0.9, np.zeros(dim - 3)],
            np.r_[1.7, -0.9, 1.0, np.zeros(dim - 3)],
        ])
        difficulties = np.array([0.10, 0.22, 0.46, 0.70])

    embeddings = np.zeros((n, dim))
    risk = np.zeros(n)
    domains = np.zeros(n, dtype=int)

    for t in range(n):
        if t < 500:
            domain = 0 if (t // 80) % 2 == 0 else 1
        elif t < 1000:
            domain = 2
        elif t < 1600:
            domain = 1 if (t // 70) % 2 == 0 else 3
        else:
            domain = 0 if (t // 60) % 3 == 0 else 3

        domains[t] = domain
        local_semantic_shift = 0.25 * np.sin(2 * np.pi * t / 210.0)
        embeddings[t] = centers[domain] + local_semantic_shift + rng.normal(scale=0.62, size=dim)
        novelty = min(1.0, np.linalg.norm(embeddings[t] - centers[0]) / 4.5)
        burst = 0.14 if 970 <= t < 1110 or 1570 <= t < 1710 else 0.0
        risk[t] = np.clip(difficulties[domain] + 0.15 * novelty + burst + rng.normal(scale=0.085), 0.0, 1.0)

    return embeddings, risk, domains


def evaluate_stream(embeddings: np.ndarray, risk: np.ndarray, alpha: float = 0.1) -> list[dict[str, float | str]]:
    n = len(risk)
    cal_start = 300
    window = 280
    recent_window = 150
    bandwidth = 1.85
    rows: list[dict[str, float | str]] = []

    for method in ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]:
        alpha_t = alpha
        misses: list[float] = []
        thresholds: list[float] = []
        ess_values: list[float] = []
        drift_values: list[float] = []

        for t in range(cal_start, n):
            if method == "static_crc":
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
                weights = np.exp(-(dist**2) / bandwidth**2) + 1e-8
                weights /= weights.sum()
                drift = float(np.average(dist, weights=weights))
            else:
                idx = np.arange(max(0, t - 4 * window), t)
                dist = np.linalg.norm(embeddings[idx] - embeddings[t], axis=1)
                sim = np.exp(-(dist**2) / bandwidth**2)
                sim /= max(sim.sum(), 1e-12)
                recent_idx = np.arange(max(0, t - recent_window), t)
                recent_mask = np.isin(idx, recent_idx).astype(float)
                recent = recent_mask / max(1.0, recent_mask.sum())
                local_drift = float(np.average(dist, weights=sim + 1e-12))
                gate = 1.0 / (1.0 + np.exp(2.2 * (local_drift - 2.05)))
                weights = gate * sim + (1.0 - gate) * recent + 1e-8
                weights /= weights.sum()
                drift = local_drift

            q = weighted_quantile(risk[idx], weights, 1.0 - alpha_t)
            miss = float(risk[t] > q)
            if method in {"adaptive_crc", "promptshift_crc"}:
                alpha_t = float(np.clip(alpha_t + 0.02 * (alpha - miss), 0.02, 0.35))
            misses.append(miss)
            thresholds.append(q)
            ess_values.append(float(1.0 / np.sum(weights**2)))
            drift_values.append(drift)

        rows.append({
            "method": method,
            "coverage": 1.0 - float(np.mean(misses)),
            "mean_threshold": float(np.mean(thresholds)),
            "mean_effective_calibration_size": float(np.mean(ess_values)),
            "mean_drift": float(np.mean(drift_values)),
        })

    return rows


def write_summary(all_rows: list[dict[str, float | str]]) -> None:
    out = RESULTS / "real_promptshift_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {out}")


def make_figure(all_rows: list[dict[str, float | str]]) -> None:
    if Image is None:
        return
    tasks = ["qa_topic_shift", "summarization_domain_shift", "safety_intent_shift"]
    methods = ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]
    labels = {
        "static_crc": "Static",
        "rolling_crc": "Rolling",
        "adaptive_crc": "Adaptive",
        "representation_weighted_crc": "Rep-weighted",
        "promptshift_crc": "PromptShift",
    }
    colors = {
        "static_crc": "#777777",
        "rolling_crc": "#2a6fbb",
        "adaptive_crc": "#1b9e77",
        "representation_weighted_crc": "#984ea3",
        "promptshift_crc": "#d95f02",
    }
    data = {(r["task"], r["method"]): float(r["coverage"]) for r in all_rows}

    width, height = 1300, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((70, 35), "LLM-style prompt-shift benchmark prototypes: coverage by method", fill="#111111", font=font)

    plot_left, plot_top, plot_right, plot_bottom = 90, 95, 1240, 640
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#d0d0d0")
    y_min, y_max = 0.25, 0.95
    for cov in [0.3, 0.5, 0.7, 0.9]:
        y = int(plot_bottom - (cov - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#eeeeee", width=1)
        draw.text((35, y - 6), f"{cov:.1f}", fill="#333333", font=font)
    target_y = int(plot_bottom - (0.9 - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
    draw.line((plot_left, target_y, plot_right, target_y), fill="#111111", width=2)

    group_w = (plot_right - plot_left) / len(tasks)
    bar_w = group_w / 7
    for gi, task in enumerate(tasks):
        base_x = plot_left + gi * group_w + bar_w
        for mi, method in enumerate(methods):
            cov = data[(task, method)]
            x0 = int(base_x + mi * bar_w)
            x1 = int(x0 + bar_w * 0.75)
            y = int(plot_bottom - (cov - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
            y = max(plot_top, min(plot_bottom, y))
            draw.rectangle((x0, y, x1, plot_bottom), fill=colors[method])
        draw.text((int(base_x), plot_bottom + 22), task.replace("_", " "), fill="#111111", font=font)

    lx, ly = 90, 680
    for method in methods:
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=colors[method])
        draw.text((lx + 25, ly + 2), labels[method], fill="#111111", font=font)
        lx += 220

    out = FIGURES / "figure2_real_promptshift_summary.png"
    img.save(out)
    print(f"Wrote {out}")


def main() -> None:
    tasks = ["qa_topic_shift", "summarization_domain_shift", "safety_intent_shift"]
    all_rows: list[dict[str, float | str]] = []
    for task in tasks:
        embeddings, risk, _ = make_task_stream(task)
        rows = evaluate_stream(embeddings, risk)
        for row in rows:
            row = {"task": task, **row}
            all_rows.append(row)
    write_summary(all_rows)
    make_figure(all_rows)


if __name__ == "__main__":
    main()
