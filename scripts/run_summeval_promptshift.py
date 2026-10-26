"""SummEval factuality-shift experiment for PromptShift-CRC."""

from __future__ import annotations

import csv
import math
import re
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


def load_rows() -> tuple[list[dict], str]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("mteb/summeval", split="test")
        return [dict(item) for item in ds], "huggingface_mteb_summeval_test"
    except Exception as exc:
        raise RuntimeError(f"Could not load SummEval: {exc}") from exc


def text_features(source: str, summary: str, system_rank: int) -> np.ndarray:
    source_words = re.findall(r"[a-z]+", source.lower())
    summary_words = re.findall(r"[a-z]+", summary.lower())
    s_n = max(len(source_words), 1)
    y_n = max(len(summary_words), 1)
    source_set = set(source_words)
    summary_set = set(summary_words)
    overlap = len(source_set & summary_set) / max(len(summary_set), 1)
    compression = y_n / s_n
    number_rate = len(re.findall(r"\d+", summary)) / y_n
    quote_rate = (summary.count("'") + summary.count('"')) / max(len(summary), 1)
    entity_shape = sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", summary)) / y_n
    return np.array(
        [
            min(s_n / 900.0, 1.0),
            min(y_n / 120.0, 1.0),
            min(compression * 8.0, 1.0),
            overlap,
            number_rate,
            quote_rate * 50.0,
            entity_shape,
            system_rank / 15.0,
            math.sin(y_n / 10.0),
            math.cos(s_n / 50.0),
        ],
        dtype=float,
    )


def build_stream(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    examples = []
    for item in rows:
        source = str(item.get("text") or "")
        summaries = item.get("machine_summaries") or []
        consistency = item.get("consistency") or []
        relevance = item.get("relevance") or []
        coherence = item.get("coherence") or []
        for j, summary in enumerate(summaries):
            if j >= len(consistency):
                continue
            cons = float(consistency[j])
            rel = float(relevance[j]) if j < len(relevance) else cons
            coh = float(coherence[j]) if j < len(coherence) else cons
            # SummEval human scores are on a 1--5 scale. We use low factual
            # consistency as risk, with small penalties for weak relevance and
            # coherence so that risk reflects usable summarization quality.
            risk = np.clip(
                0.78 * (1.0 - (cons - 1.0) / 4.0)
                + 0.12 * (1.0 - (rel - 1.0) / 4.0)
                + 0.10 * (1.0 - (coh - 1.0) / 4.0),
                0.0,
                1.0,
            )
            feat = text_features(source, str(summary), j)
            source_len = len(re.findall(r"[a-z]+", source.lower()))
            examples.append(
                {
                    "features": feat,
                    "risk": float(risk),
                    "source_len": source_len,
                    "system_rank": j,
                    "consistency": cons,
                }
            )

    # Build a deployment-like stream: short/easy news articles first, then
    # longer and lower-consistency regimes, then a mixed return. This gives
    # the calibration layer a real source-domain and model-quality shift.
    examples.sort(key=lambda x: (x["source_len"], -x["consistency"], x["system_rank"]))
    n = len(examples)
    easy = examples[: n // 3]
    mixed = examples[n // 3 : 2 * n // 3]
    hard = examples[2 * n // 3 :]
    stream = easy + hard + mixed  # revision: the repeated half of `hard` is removed (no item appears twice)

    embeddings = np.vstack([x["features"] for x in stream])
    risk = np.array([x["risk"] for x in stream], dtype=float)
    return embeddings, risk


def evaluate(embeddings: np.ndarray, risk: np.ndarray, alpha: float = 0.1) -> list[dict[str, float | str]]:
    n = len(risk)
    cal_start = min(320, max(120, n // 5))
    window = min(260, max(100, n // 6))
    recent_window = min(120, max(60, n // 12))
    bandwidth = 0.75
    rows = []
    for method in ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"]:
        alpha_t = alpha
        misses = []
        thresholds = []
        ess_values = []
        drift_values = []
        for t in range(cal_start, n):
            if method == "static_crc":
                idx = np.arange(0, cal_start)
                weights = np.ones(len(idx)) / len(idx)
                drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
            elif method in {"rolling_crc", "adaptive_crc"}:
                idx = np.arange(max(0, t - window), t)
                weights = np.ones(len(idx)) / len(idx)
                drift = float(np.linalg.norm(embeddings[t] - embeddings[idx].mean(axis=0)))
            else:
                idx = np.arange(max(0, t - 3 * window), t)
                dist = np.linalg.norm(embeddings[idx] - embeddings[t], axis=1)
                sim = np.exp(-(dist**2) / bandwidth**2) + 1e-8
                sim /= sim.sum()
                drift = float(np.average(dist, weights=sim))
                if method == "representation_weighted_crc":
                    weights = sim
                else:
                    recent_idx = np.arange(max(0, t - recent_window), t)
                    recent_mask = np.isin(idx, recent_idx).astype(float)
                    recent = recent_mask / max(1.0, recent_mask.sum())
                    gate = 1.0 / (1.0 + np.exp(4.0 * (drift - 0.85)))
                    weights = gate * sim + (1.0 - gate) * recent + 1e-8
                    weights /= weights.sum()
            q = weighted_quantile(risk[idx], weights, 1.0 - alpha_t)
            miss = float(risk[t] > q)
            if method in {"adaptive_crc", "promptshift_crc"}:
                alpha_t = float(np.clip(alpha_t + 0.018 * (alpha - miss), 0.02, 0.35))
            misses.append(miss)
            thresholds.append(q)
            ess_values.append(float(1.0 / np.sum(weights**2)))
            drift_values.append(drift)
        coverage = 1.0 - float(np.mean(misses))
        rows.append(
            {
                "method": method,
                "coverage": coverage,
                "mean_threshold": float(np.mean(thresholds)),
                "mean_effective_calibration_size": float(np.mean(ess_values)),
                "mean_drift": float(np.mean(drift_values)),
                "absolute_coverage_error": abs(coverage - 0.9),
            }
        )
    return rows


def write_summary(rows: list[dict[str, float | str]], source: str) -> None:
    out = RESULTS / "summeval_promptshift_summary.csv"
    fieldnames = ["source"] + list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"source": source, **row})
    print(f"Wrote {out}")


def make_figure(rows: list[dict[str, float | str]]) -> None:
    if Image is None:
        return
    width, height = 1100, 650
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((65, 35), "SummEval factuality-shift experiment: coverage by method", fill="#111111", font=font)
    labels = {
        "static_crc": "Static",
        "rolling_crc": "Rolling",
        "adaptive_crc": "Adaptive",
        "representation_weighted_crc": "Rep-weighted",
        "promptshift_crc": "PromptShift",
    }
    colors = ["#777777", "#2a6fbb", "#1b9e77", "#984ea3", "#d95f02"]
    plot_left, plot_top, plot_right, plot_bottom = 90, 90, 1020, 520
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#d0d0d0")
    y_min, y_max = 0.65, 1.0
    for cov in [0.7, 0.8, 0.9, 1.0]:
        y = int(plot_bottom - (cov - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#eeeeee", width=1)
        draw.text((40, y - 6), f"{cov:.1f}", fill="#333333", font=font)
    target_y = int(plot_bottom - (0.9 - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
    draw.line((plot_left, target_y, plot_right, target_y), fill="#111111", width=2)
    bar_w = (plot_right - plot_left) / (len(rows) * 1.6)
    for i, row in enumerate(rows):
        cov = float(row["coverage"])
        x0 = int(plot_left + i * bar_w * 1.6 + 25)
        x1 = int(x0 + bar_w)
        y = int(plot_bottom - (cov - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        y = max(plot_top, min(plot_bottom, y))
        draw.rectangle((x0, y, x1, plot_bottom), fill=colors[i])
        draw.text((x0 - 4, y - 18), f"{cov:.3f}", fill="#111111", font=font)
        draw.text((x0 - 8, plot_bottom + 22), labels[str(row["method"])], fill="#111111", font=font)
    out = FIGURES / "figure6_summeval_promptshift.png"
    img.save(out)
    print(f"Wrote {out}")


def main() -> None:
    rows_raw, source = load_rows()
    embeddings, risk = build_stream(rows_raw)
    rows = evaluate(embeddings, risk)
    write_summary(rows, source)
    make_figure(rows)


if __name__ == "__main__":
    main()
