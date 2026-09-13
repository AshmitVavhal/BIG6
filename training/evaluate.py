#!/usr/bin/env python3
"""
SATQUERY AI - GeoChat Model Evaluation & Comparative Benchmark
Evaluates GeoChat on held-out test sets.
Generates objective metrics (ROUGE, BLEU) and an interactive human-grading HTML report.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root and backend dir are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
for p in [str(BASE_DIR), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from app.models.geochat_model import GeoChatModelWrapper

def compute_nlp_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE and BLEU scores across candidate predictions and ground-truth references."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1

    r1_list, r2_list, rl_list = [], [], []
    bleu1_list, bleu4_list = [], []

    for pred, ref in zip(predictions, references):
        # ROUGE
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)

        # BLEU
        ref_tokens = [ref.lower().split()]
        pred_tokens = pred.lower().split()
        
        b1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        b4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu1_list.append(b1)
        bleu4_list.append(b4)

    return {
        "rouge1": round(float(sum(r1_list) / max(len(r1_list), 1) * 100), 2),
        "rouge2": round(float(sum(r2_list) / max(len(r2_list), 1) * 100), 2),
        "rougeL": round(float(sum(rl_list) / max(len(rl_list), 1) * 100), 2),
        "bleu1": round(float(sum(bleu1_list) / max(len(bleu1_list), 1) * 100), 2),
        "bleu4": round(float(sum(bleu4_list) / max(len(bleu4_list), 1) * 100), 2)
    }

def generate_html_report(
    eval_data: List[Dict[str, Any]],
    geochat_metrics: Dict[str, float],
    output_html_path: Path
):
    """Generate an interactive HTML human-grading report."""
    rows_html = []
    for idx, item in enumerate(eval_data, 1):
        cat = item.get("category", "General")
        q = item.get("question", "")
        gt = item.get("ground_truth", "")
        ans = item.get("geochat_answer", "")
        img_name = item.get("image_name", "")

        row = f"""
        <div class="eval-card" id="sample-{idx}">
            <div class="card-header">
                <span class="badge category-badge">{cat.upper()}</span>
                <span class="sample-id">Sample #{idx} &mdash; <code>{img_name}</code></span>
            </div>
            <div class="card-body">
                <div class="query-section">
                    <strong>Question:</strong> {q}
                </div>
                <div class="ground-truth">
                    <strong>Ground Truth:</strong> {gt}
                </div>
                <div class="answers-grid">
                    <div class="model-column geochat-col">
                        <h4>GeoChat Remote-Sensing VLM</h4>
                        <div class="answer-box">{ans}</div>
                        <div class="grading-group">
                            <label><input type="radio" name="grade_geochat_{idx}" value="correct"> Correct</label>
                            <label><input type="radio" name="grade_geochat_{idx}" value="partial"> Partial</label>
                            <label><input type="radio" name="grade_geochat_{idx}" value="incorrect"> Incorrect</label>
                        </div>
                    </div>
                </div>
                <div class="notes-section">
                    <input type="text" placeholder="Analyst comments / Domain accuracy notes..." class="notes-input" />
                </div>
            </div>
        </div>
        """
        rows_html.append(row)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SatQuery AI &mdash; GeoChat Evaluation Report</title>
    <style>
        :root {{
            --bg-color: #0b1120;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #00ffcc;
            --accent-amber: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .metrics-summary {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-box {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .eval-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            overflow: hidden;
        }}
        .card-header {{
            background: #0f172a;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }}
        .card-body {{ padding: 16px; }}
        .badge {{
            font-size: 11px;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
            background: var(--accent-cyan);
            color: #0b1120;
        }}
        .query-section, .ground-truth {{ margin-bottom: 12px; font-size: 14px; }}
        .ground-truth {{ color: var(--accent-amber); }}
        .answers-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
            margin-top: 12px;
        }}
        .model-column {{
            background: #0f172a;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}
        .geochat-col h4 {{ color: var(--accent-cyan); margin: 0 0 8px 0; }}
        .answer-box {{ font-size: 13px; line-height: 1.5; min-height: 60px; }}
        .grading-group {{ margin-top: 12px; font-size: 12px; display: flex; gap: 12px; }}
        .notes-input {{
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 8px;
            color: var(--text-main);
            font-size: 12px;
            box-sizing: border-box;
            margin-top: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SATQUERY AI &mdash; GeoChat Evaluation Report</h1>
        <p style="color: var(--text-muted); margin: 4px 0 0 0;">
            Benchmark evaluation for MBZUAI/GeoChat on held-out remote sensing test data.
        </p>
    </div>

    <div class="metrics-summary">
        <div class="metric-box" style="border-color: rgba(0,255,204,0.4);">
            <h3 style="color: var(--accent-cyan);">GeoChat (MBZUAI/GeoChat-7B)</h3>
            <p>ROUGE-1: <strong>{geochat_metrics.get("rouge1", 0.0)}%</strong> | ROUGE-L: <strong>{geochat_metrics.get("rougeL", 0.0)}%</strong></p>
            <p>BLEU-1: <strong>{geochat_metrics.get("bleu1", 0.0)}%</strong> | BLEU-4: <strong>{geochat_metrics.get("bleu4", 0.0)}%</strong></p>
        </div>
    </div>

    <h2>Test Set Predictions & Human Review ({len(eval_data)} samples)</h2>
    {"".join(rows_html)}
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Human Evaluation HTML Report written to: {output_html_path}")

def evaluate(args):
    base_dir = Path(__file__).resolve().parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else base_dir / args.config
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8")) if config_path.exists() else {}

    test_file = Path(args.test_path or cfg.get("dataset", {}).get("test_path", "data/finetuning/test.jsonl"))
    if not test_file.is_absolute():
        test_file = base_dir / test_file

    if not test_file.exists():
        print(f"[ERROR] Held-out test set not found: {test_file}")
        sys.exit(1)

    out_dir = Path(args.output_dir or cfg.get("evaluation", {}).get("output_dir", "evaluation"))
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test samples
    with open(test_file, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip()]

    print("=" * 70)
    print(f" SATQUERY AI - GEOCHAT EVALUATION & BENCHMARK SUITE ({len(test_samples)} Test Samples)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapper = GeoChatModelWrapper(device=device)

    results = []
    preds = []
    ground_truths = []
    latencies = []

    for idx, sample in enumerate(test_samples, 1):
        img_p = Path(sample["image"])
        if not img_p.is_absolute():
            img_p = base_dir / img_p
        q = sample["question"]
        gt = sample["answer"]
        ground_truths.append(gt)

        pil_img = Image.open(img_p).convert("RGB")
        img_rgb = np.array(pil_img)

        ans, caption, latency = wrapper.generate_vqa_answer(img_rgb, q)
        preds.append(ans)
        latencies.append(latency)

        results.append({
            "id": idx,
            "image": sample["image"],
            "image_name": img_p.name,
            "category": sample.get("category", "General"),
            "question": q,
            "ground_truth": gt,
            "geochat_answer": ans,
            "latency_ms": round(latency, 1)
        })
        print(f"  [{idx}/{len(test_samples)}] {img_p.name} -> {latency:.1f}ms")

    metrics = compute_nlp_metrics(preds, ground_truths)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1)

    print("\n" + "=" * 70)
    print(" GEOCHAT EVALUATION METRICS")
    print("=" * 70)
    print(f"  * ROUGE-1:         {metrics['rouge1']}%")
    print(f"  * ROUGE-2:         {metrics['rouge2']}%")
    print(f"  * ROUGE-L:         {metrics['rougeL']}%")
    print(f"  * BLEU-1:          {metrics['bleu1']}%")
    print(f"  * BLEU-4:          {metrics['bleu4']}%")
    print(f"  * Mean Latency:    {metrics['mean_latency_ms']} ms")
    print("=" * 70 + "\n")

    # Save results JSON
    res_file = out_dir / "geochat_results.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": results}, f, indent=2)
    print(f"[OK] Evaluation results saved to: {res_file}")

    # Generate HTML Report
    html_path = out_dir / "evaluation_report.html"
    generate_html_report(results, metrics, html_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoChat on SatQuery Test Set")
    parser.add_argument("--config", default="training/config.yaml", help="Path to YAML config")
    parser.add_argument("--test_path", default=None, help="Path to test.jsonl")
    parser.add_argument("--output_dir", default="evaluation", help="Output directory for reports")
    args = parser.parse_args()

    evaluate(args)
