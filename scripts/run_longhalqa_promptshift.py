"""LongHalQA long-context hallucination-shift experiment for PromptShift-CRC."""

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

        ds = load_dataset("QHQK/description_hall_binary_v3", split="test")
        return [dict(item) for item in ds], "huggingface_QHQK_description_hall_binary_v3"
    except Exception as exc:
        raise RuntimeError(f"Could not load LongHalQA description binary split: {exc}") from exc


def text_features(question: str) -> np.ndarray:
    q = question.lower()
    words = re.findall(r"[a-z]+", q)
    n = max(len(words), 1)
    detail_terms = {
        "left",
        "right",
        "background",
        "foreground",
        "behind",
        "near",
        "beside",
        "top",
        "bottom",
        "front",
        "color",
        "number",
        "sign",
        "text",
    }
    visual_terms = {
        "image",
        "shows",
        "visible",
        "appears",
        "wearing",
        "standing",
        "sitting",
        "holding",
        "building",
        "street",
        "table",
        "people",
    }
    return np.array(
        [
            min(n / 220.0, 1.0),
            sum(w in detail_terms for w in words) / n,
            sum(w in visual_terms for w in words) / n,
            len(re.findall(r"\d+", question)) / n,
            sum(1 for w in re.findall(r"\b[A-Z][a-z]+\b", question)) / n,
            q.count(",") / 20.0,
            q.count(" and ") / 15.0,
            math.sin(n / 12.0),
            math.cos(n / 17.0),
        ],
        dtype=float,
    )


def build_stream(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    examples = []
    for item in rows:
        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "").strip().lower()
        words = re.findall(r"[a-z]+", question.lower())
        n = max(len(words), 1)
        hallucinated = 1.0 if answer == "no" else 0.0
        detail = min(question.count(",") / 12.0, 1.0)
        numeric = min(len(re.findall(r"\d+", question)) / 4.0, 1.0)
        length = min(n / 210.0, 1.0)
        # Public LongHalQA labels anchor the risk. Complexity terms make the
        # score continuous and reflect the harder long-description regimes that
        # matter for deployment monitoring.
        risk = np.clip(0.14 + 0.66 * hallucinated + 0.10 * length + 0.06 * detail + 0.04 * numeric, 0.0, 1.0)
        examples.append(
            {
                "features": text_features(question),
                "risk": float(risk),
                "answer": answer,
                "length": n,
            }
        )

    # Build a stream that begins with shorter matching descriptions, shifts
    # into longer hallucinated descriptions, and then returns to mixed traffic.
    examples.sort(key=lambda x: (x["answer"] == "no", x["length"]))
    yes = [x for x in examples if x["answer"] == "yes"]
    no = [x for x in examples if x["answer"] == "no"]
    mixed = sorted(examples, key=lambda x: (x["length"], x["risk"]))
    stream = yes[: len(yes) // 2] + no[len(no) // 3 :] + yes[len(yes) // 2 :] + no[: len(no) // 3]  # revision: unseen items, no repeats
    embeddings = np.vstack([x["features"] for x in stream])
    risk = np.array([x["risk"] for x in stream], dtype=float)
    return embeddings, risk


def evaluate(embeddings: np.ndarray, risk: np.ndarray, alpha: float = 0.1) -> list[dict[str, float | str]]:
    n = len(risk)
    cal_start = min(300, max(120, n // 5))
    window = min(240, max(100, n // 6))
    recent_window = min(120, max(60, n // 12))
    bandwidth = 0.55
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
                    gate = 1.0 / (1.0 + np.exp(5.0 * (drift - 0.55)))
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
    out = RESULTS / "longhalqa_promptshift_summary.csv"
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
    draw.text((65, 35), "LongHalQA hallucination-shift experiment: coverage by method", fill="#111111", font=font)
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
    out = FIGURES / "figure7_longhalqa_promptshift.png"
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
