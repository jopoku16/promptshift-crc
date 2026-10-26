"""RealToxicityPrompts safety-shift experiment for PromptShift-CRC."""

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


def load_rows(limit: int = 5000) -> tuple[list[dict], str]:
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("allenai/real-toxicity-prompts", split=f"train[:{limit}]")
        return [dict(item) for item in ds], "huggingface_allenai_real_toxicity_prompts"
    except Exception as exc:
        raise RuntimeError(f"Could not load RealToxicityPrompts: {exc}") from exc


def text_features(text: str) -> np.ndarray:
    q = text.lower()
    words = re.findall(r"[a-z]+", q)
    n = max(len(words), 1)
    toxic_lexicon = {"hate", "kill", "stupid", "idiot", "sex", "racist", "violent", "attack", "war", "abuse"}
    identity = {"woman", "man", "black", "white", "muslim", "christian", "gay", "jewish", "immigrant"}
    profanity_shape = sum(1 for w in words if len(w) > 10) / n
    return np.array(
        [
            min(n / 40.0, 1.0),
            sum(w in toxic_lexicon for w in words) / n,
            sum(w in identity for w in words) / n,
            profanity_shape,
            q.count("!") / 3.0,
            q.count("?") / 3.0,
            math.sin(n),
            math.cos(n),
        ],
        dtype=float,
    )


def build_stream(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # Sort into a stream that moves from lower prompt toxicity into challenging
    # and higher-toxicity regions, then returns to mixed traffic.
    def prompt_tox(item: dict) -> float:
        return float(item.get("prompt", {}).get("toxicity") or 0.0)

    ordered = sorted(rows, key=lambda item: (bool(item.get("challenging")), prompt_tox(item)))
    low = ordered[: len(ordered) // 3]
    mid = ordered[len(ordered) // 3 : 2 * len(ordered) // 3]
    high = ordered[2 * len(ordered) // 3 :]
    stream = low + high + mid + high[: len(high) // 2]
    stream = stream[: min(len(stream), 4200)]

    embeddings = []
    risk = []
    for item in stream:
        prompt = item.get("prompt", {})
        continuation = item.get("continuation", {})
        text = str(prompt.get("text") or "")
        ptox = float(prompt.get("toxicity") or 0.0)
        ctox = float(continuation.get("toxicity") or 0.0)
        severe = float(continuation.get("severe_toxicity") or 0.0)
        threat = float(continuation.get("threat") or 0.0)
        insult = float(continuation.get("insult") or 0.0)
        x = text_features(text)
        x = np.r_[x, ptox, float(item.get("challenging", False))]
        embeddings.append(x)
        # Risk is continuation toxicity, made slightly more sensitive to severe
        # toxicity, threat, and insult. This uses public dataset scores.
        risk.append(np.clip(0.70 * ctox + 0.12 * severe + 0.10 * threat + 0.08 * insult, 0.0, 1.0))
    return np.vstack(embeddings), np.array(risk, dtype=float)


def evaluate(embeddings: np.ndarray, risk: np.ndarray, alpha: float = 0.1) -> list[dict[str, float | str]]:
    n = len(risk)
    cal_start = min(600, max(160, n // 5))
    window = min(500, max(150, n // 6))
    recent_window = min(200, max(80, n // 12))
    bandwidth = 0.9
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
                    gate = 1.0 / (1.0 + np.exp(3.0 * (drift - 0.95)))
                    weights = gate * sim + (1.0 - gate) * recent + 1e-8
                    weights /= weights.sum()
            q = weighted_quantile(risk[idx], weights, 1.0 - alpha_t)
            miss = float(risk[t] > q)
            if method in {"adaptive_crc", "promptshift_crc"}:
                alpha_t = float(np.clip(alpha_t + 0.015 * (alpha - miss), 0.02, 0.35))
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
    out = RESULTS / "realtoxicity_promptshift_summary.csv"
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
    draw.text((65, 35), "RealToxicityPrompts safety-shift experiment: coverage by method", fill="#111111", font=font)
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
    out = FIGURES / "figure5_realtoxicity_promptshift.png"
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

