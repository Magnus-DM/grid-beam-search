"""Download the evaluation sentences and, optionally, the corpus shard for PMI statistics.

Evaluation set: the first N English-French pairs of the training split of
thesistranslation/distilled-ccmatrix-fr-en on Hugging Face, written to data/eval.tsv.

PMI corpus (--pmi-corpus): shard 1 of the same dataset (about 330 MB), written to
data/ccmatrix/. Statistics are counted on its first 300,000 pairs, so they never
include the evaluation sentences, which come from shard 0.
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

DATASET = "thesistranslation/distilled-ccmatrix-fr-en"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PMI_SHARD = "data/train-00001-of-00016-adc4597968b1c21c.parquet"
DATA_DIR = Path("data")


def download_eval_set(n: int, path: Path) -> None:
    rows = []
    for offset in range(0, n, 100):
        length = min(100, n - offset)
        url = (
            f"{ROWS_API}?dataset={DATASET}&config=default&split=train"
            f"&offset={offset}&length={length}"
        )
        with urllib.request.urlopen(url) as response:
            rows += [r["row"]["translation"] for r in json.load(response)["rows"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["en", "fr"])
        for row in rows:
            writer.writerow([row["en"], row["fr"]])
    print(f"Wrote {len(rows)} sentence pairs to {path}")


def download_pmi_corpus(target_dir: Path) -> None:
    target = target_dir / Path(PMI_SHARD).name
    if target.exists():
        print(f"{target} already exists")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{PMI_SHARD}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, target)
    print(f"Saved {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of evaluation pairs")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "eval.tsv")
    parser.add_argument("--pmi-corpus", action="store_true", help="also download the PMI shard")
    args = parser.parse_args()

    download_eval_set(args.n, args.out)
    if args.pmi_corpus:
        download_pmi_corpus(DATA_DIR / "ccmatrix")


if __name__ == "__main__":
    main()
