"""Re-score the translations from the original run (results/outputs/) against references.

Prints the reported scores (results/scores.csv) next to scores recomputed from the stored
translations and the references in --data. The reported scores used the course's copy of
the 200 sentence pairs; see "Data" in the README for how it differs from the raw dataset.
"""

import argparse
import csv
import json
from pathlib import Path

from gbs.baseline import evaluate
from run_benchmark import load_eval_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval.tsv"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    references = [row["fr"] for row in load_eval_set(args.data)]
    with open(args.results / "scores.csv", newline="") as f:
        reported = list(csv.DictReader(f))

    print(f"{'method':<12} {'width':>5}  {'chrF reported':>13}  {'chrF recomputed':>15}")
    for row in reported:
        path = args.results / "outputs" / f"{row['method']}_bw={row['beam_width']}.json"
        hypotheses = json.loads(path.read_text(encoding="utf-8"))
        recomputed = evaluate(hypotheses, references).chrf
        print(f"{row['method']:<12} {row['beam_width']:>5}  {row['chrf']:>13}  {recomputed:>15.2f}")


if __name__ == "__main__":
    main()
