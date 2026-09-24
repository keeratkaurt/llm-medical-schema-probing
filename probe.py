"""Contrastive source-associated symptom probe; not a clinical decision system."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "patient": "A patient with {disease} often experiences",
    "symptom": "A common symptom of {disease} is",
}

def score_continuation(model, tokenizer, prefix, continuation):
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + continuation, add_special_tokens=False).input_ids
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f"Token boundary changed for {prefix!r}")
    x = torch.tensor([full_ids])
    with torch.inference_mode():
        logits = model(x).logits[0, :-1]
        log_probs = torch.log_softmax(logits, dim=-1)
    positions = range(len(prefix_ids), len(full_ids))
    values = [float(log_probs[i - 1, full_ids[i]]) for i in positions]
    return -sum(values) / (len(values) * math.log(2))

def run(data_path, model_name, output_dir):
    rows = list(csv.DictReader(data_path.open()))
    diseases = list(dict.fromkeys(r["disease"] for r in rows))
    assert len(diseases) == 4 and len(rows) == 24
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    details = []
    for prompt_name, template in PROMPTS.items():
        for row in rows:
            scores = {disease: score_continuation(model, tokenizer,
                template.format(disease=disease), " " + row["symptom"] + ".")
                for disease in diseases}
            ranking = sorted(diseases, key=lambda d: scores[d])
            details.append({**row, "prompt": prompt_name, "predicted": ranking[0],
                "source_rank": ranking.index(row["disease"]) + 1,
                **{f"bits_per_token__{d}": scores[d] for d in diseases}})
            print(prompt_name, row["disease"], row["symptom"], "=>", ranking[0], flush=True)
    with (output_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    metrics = {"model": model_name, "n_symptoms": len(rows), "n_diseases": len(diseases),
        "chance_top1": 1 / len(diseases),
        "interpretation": "Source-associated ranking; other diseases are not clinical negatives."}
    for prompt_name in PROMPTS:
        sub = [r for r in details if r["prompt"] == prompt_name]
        correct = np.array([r["source_rank"] == 1 for r in sub], dtype=int)
        rng = np.random.default_rng(2026)
        boot = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(10000)]
        metrics[prompt_name] = {"top1": float(correct.mean()), "top1_count": int(correct.sum()),
            "mean_source_rank": float(np.mean([r["source_rank"] for r in sub])),
            "bootstrap_95pct_interval": [float(x) for x in np.quantile(boot, [.025, .975])],
            "per_disease": {d: f"{sum(r['source_rank'] == 1 for r in sub if r['disease'] == d)}/{sum(r['disease'] == d for r in sub)}" for d in diseases}}
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert/distilgpt2")
    parser.add_argument("--data", type=Path, default=Path("data/symptoms.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data, args.model, args.output)
