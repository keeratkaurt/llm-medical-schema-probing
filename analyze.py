"""Audit existing source-associated symptom scores without rerunning a model."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def analyze(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No score rows")
    score_columns = [c for c in rows[0] if c.startswith("bits_per_token__")]
    diseases = [c.removeprefix("bits_per_token__") for c in score_columns]
    if len(diseases) < 2:
        raise ValueError("Need scores for at least two diseases")
    prompts = sorted({r["prompt"] for r in rows})
    grouped = defaultdict(dict)
    confusion = defaultdict(Counter)
    errors = []
    for row in rows:
        source = row["disease"]
        if source not in diseases or row["prompt"] in grouped[(source, row["symptom"])]:
            raise ValueError("Unknown source disease or duplicate phrase/prompt")
        scores = {d: float(row[f"bits_per_token__{d}"]) for d in diseases}
        if not all(math.isfinite(x) for x in scores.values()):
            raise ValueError("Non-finite score")
        ordered = sorted(diseases, key=lambda d: scores[d])
        if row["predicted"] != ordered[0] or int(row["source_rank"]) != ordered.index(source) + 1:
            raise ValueError("Stored predicted/source rank does not match raw scores")
        grouped[(source, row["symptom"])][row["prompt"]] = ordered[0]
        confusion[row["prompt"]][(source, ordered[0])] += 1
        if ordered[0] != source:
            errors.append({"prompt": row["prompt"], "source": source,
                           "symptom": row["symptom"], "predicted": ordered[0],
                           "source_rank": int(row["source_rank"]),
                           "source_minus_best_bits_per_token": round(scores[source] - scores[ordered[0]], 5)})
    if any(set(d) != set(prompts) for d in grouped.values()):
        raise ValueError("Every phrase must have every prompt")
    predictions_changed = sum(len(set(d.values())) > 1 for d in grouped.values())
    source_match_changed = sum(len({p == source for p in d.values()}) > 1
                               for (source, _), d in grouped.items())
    return {"n_phrases": len(grouped), "n_prompts": len(prompts),
            "predictions_changed_across_prompts": predictions_changed,
            "source_match_changed_across_prompts": source_match_changed,
            "confusion": {prompt: {source: {pred: confusion[prompt][(source, pred)]
                                            for pred in diseases} for source in diseases}
                          for prompt in prompts},
            "errors": sorted(errors, key=lambda e: (e["prompt"], -e["source_minus_best_bits_per_token"]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.scores.open(newline="") as stream:
        report = analyze(csv.DictReader(stream))
    formatted = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted)
    else:
        print(formatted, end="")
