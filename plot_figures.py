"""Bar charts of BLEU, chrF and chrF++ per beam width from a scores.csv file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABELS = {
    "beam": "Beam search",
    "grid_random": "Grid beam, random",
    "grid_pmi": "Grid beam, PMI",
}
METRICS = {"bleu": "BLEU", "chrf": "chrF", "chrf_pp": "chrF++"}


def plot_scores(scores_csv: Path, out_dir: Path, methods) -> None:
    df = pd.read_csv(scores_csv)
    df = df[df["method"].isin(methods)].replace({"method": LABELS})
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in METRICS.items():
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(df, x="beam_width", y=metric, hue="method", palette="Set2")
        ax.set(
            title=f"{label} by beam width",
            xlabel="Beam width",
            ylabel=label,
            ylim=(0, 100),
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9, padding=2)
        plt.legend(title=None, loc="upper center", ncol=3, frameon=False)
        plt.tight_layout()
        path = out_dir / f"{metric}.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("results/scores.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs"))
    parser.add_argument("--methods", nargs="+", default=list(LABELS))
    args = parser.parse_args()
    plot_scores(args.scores, args.out, args.methods)


if __name__ == "__main__":
    main()
