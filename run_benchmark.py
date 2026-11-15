"""Compare beam search with grid beam search on English-to-French translation with T5-small.

Methods:
  beam         ordinary beam search (no constraints)
  grid_random  grid beam search with 1-4 random words or word pairs from the reference
  grid_pmi     grid beam search with constraints picked from the source by PMI

Translations are cached per method and beam width in --out/outputs/, so an interrupted
run resumes where it stopped. Scores are written to --out/scores.csv.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from gbs.baseline import beam_search, detect_device, evaluate
from gbs.grid_beam_search import grid_beam_search
from gbs.pmi_constraints import PMIConstraintSelector
from gbs.random_constraints import extract_random_constraints

PROMPT = "translate English to French: "
CONSTRAINT_SEED = 42
METHODS = ("beam", "grid_random", "grid_pmi")


def count_tokens(text, tokenizer):
    """Number of tokens without special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def load_eval_set(path, n_sentences=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[:n_sentences] if n_sentences else rows


def load_pmi_selector(stats_file, corpus_dir):
    selector = PMIConstraintSelector(min_count=5, window_size=5)
    if os.path.exists(stats_file):
        selector.load_statistics(stats_file)
    else:
        print("Computing PMI statistics from the corpus (first run only, takes a few minutes)...")
        selector.load_parallel_corpus(parquet_dir=corpus_dir, max_samples=300_000, sample_every=1)
        selector.save_statistics(stats_file)
    return selector


def translate(rows, constraints, method, beam_width, model, tokenizer, device):
    """Translate every row; the length limit is twice the English token count."""
    start_token_id = tokenizer.pad_token_id
    end_token_id = tokenizer.eos_token_id
    token_ids = []
    for i, row in enumerate(tqdm.tqdm(rows, desc=f"{method}, beam width {beam_width}")):
        src_ids = tokenizer.encode(PROMPT + row["en"], return_tensors="pt").to(device)
        max_length = count_tokens(row["en"], tokenizer) * 2
        common = dict(
            tokenizer=tokenizer,
            model=model,
            src_input_ids=src_ids,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            device=device,
            beam_width=beam_width,
            temperature=1.0,
            max_length=max_length,
            verbose=False,
        )
        if method == "beam":
            best = beam_search(**common)[0]
        else:
            best = grid_beam_search(constraints=constraints[i], **common)[0]
        token_ids.append(best.ids())
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval.tsv"))
    parser.add_argument("--n-sentences", type=int, default=None, help="use only the first N")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[2, 3, 4, 5, 10])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--out", type=Path, default=Path("runs/full"))
    parser.add_argument("--pmi-stats", default="data/pmi_statistics.pkl")
    parser.add_argument("--pmi-corpus", default="data/ccmatrix")
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = load_eval_set(args.data, args.n_sentences)
    references = [row["fr"] for row in rows]

    constraints = {
        "beam": None,
        "grid_random": [
            extract_random_constraints(r["fr"], max_num_constraints=4, seed=CONSTRAINT_SEED)
            for r in rows
        ],
    }
    if "grid_pmi" in args.methods:
        selector = load_pmi_selector(args.pmi_stats, args.pmi_corpus)
        constraints["grid_pmi"] = [
            selector.get_constraints_from_source(
                source_text=r["en"],
                max_constraints=4,
                min_pmi_threshold=0.0,
                prefer_bigrams=True,
            )
            for r in rows
        ]

    print("Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    device = detect_device()
    model.to(device).eval()

    outputs_dir = args.out / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for beam_width in args.beam_widths:
        for method in args.methods:
            cache = outputs_dir / f"{method}_bw={beam_width}.json"
            if cache.exists():
                hypotheses = json.loads(cache.read_text(encoding="utf-8"))
            else:
                hypotheses = translate(
                    rows,
                    constraints[method],
                    method,
                    beam_width,
                    model,
                    tokenizer,
                    device,
                )
                cache.write_text(json.dumps(hypotheses, ensure_ascii=False, indent=1), "utf-8")
            result = evaluate(hypotheses, references)
            scores.append(
                {
                    "method": method,
                    "beam_width": beam_width,
                    "bleu": round(result.bleu, 2),
                    "chrf": round(result.chrf, 2),
                    "chrf_pp": round(result.chrf_pp, 2),
                }
            )
            print(scores[-1])

    with open(args.out / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    print(f"Scores written to {args.out / 'scores.csv'}")


if __name__ == "__main__":
    main()
