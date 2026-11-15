"""
PMI-based Constraint Selection for Grid Beam Search

Selects French (target) constraints using ONLY the English source sentence, from
PMI(source word, target word) counted over sentence pairs of a parallel corpus. Target-side
bigram PMI is used to merge adjacent picks into two-word constraints, and a diversity
filter drops near-duplicate constraints.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# Similarity / diversity filtering helpers
# =========================


def _norm_constraint(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    # remove punctuation except apostrophes inside words
    s = re.sub(r"[^\w\s']+", "", s)
    return s


def _too_similar(a: str, b: str, ratio_thresh: float = 0.88, prefix_len: int = 4) -> bool:
    a_n = _norm_constraint(a)
    b_n = _norm_constraint(b)
    if a_n == b_n:
        return True
    # quick prefix heuristic (catches plural/suffix variants etc.)
    if len(a_n) >= prefix_len and len(b_n) >= prefix_len and a_n[:prefix_len] == b_n[:prefix_len]:
        return True
    # generic similarity ratio
    sim = SequenceMatcher(None, a_n, b_n).ratio()
    return sim >= ratio_thresh


# =========================
# Main class
# =========================


class PMIConstraintSelector:
    """
    Select lexical constraints based on PMI statistics computed from a parallel corpus.
    """

    def __init__(self, min_count: int = 5, window_size: int = 5):
        self.min_count = int(min_count)
        self.window_size = int(window_size)

        # ---- Target-side stats ----
        self.word_counts: Counter[str] = Counter()  # target word frequencies
        self.pair_counts: dict[tuple[str, str], int] = defaultdict(
            int
        )  # target co-occurrence within window
        self.bigram_counts: Counter[str] = Counter()  # target consecutive bigram frequencies
        self.total_words: int = 0
        self.total_pairs: int = 0

        self.pmi_scores: dict[tuple[str, str], float] = {}
        self.bigram_pmi_scores: dict[str, float] = {}

        # ---- Source-side + Source↔Target stats (test-time) ----
        self.src_word_counts: Counter[str] = Counter()
        self.total_src_words: int = 0

        # (src_word, tgt_word) -> joint count (sentence-pair co-occurrence)
        self.st_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.total_st_pairs: int = 0

        # Optional speed index: src -> [(tgt, joint_count), ...]
        self.src_to_tgt: dict[str, list[tuple[str, int]]] | None = None

    # =========================
    # Tokenization
    # =========================
    def _tokenize(self, text: str) -> list[str]:
        """
        Simple word tokenization:
        - lowercase
        - keep apostrophes when inside words (c'est, l'arbre)
        """
        text_lower = text.lower()
        return re.findall(r"\b\w+'\w+\b|\b\w+’\w+\b|\b\w+\b", text_lower)

    # =========================
    # Corpus processing
    # =========================
    def load_parallel_corpus(
        self,
        parquet_dir: str,
        max_samples: int | None = None,
        sample_every: int = 1,
        batch_size: int = 1000,
        skip_filename: str | None = "train-00000-of-00016-95c783ab11814c21.parquet",
    ) -> None:
        """
        Load parallel corpus from parquet files and compute statistics.
        Expects each row to contain a dict column 'translation' with keys like 'en' and 'fr'.
        """
        parquet_path = Path(parquet_dir)
        parquet_files = sorted(parquet_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")

        print(f"Loading parallel corpus from {parquet_dir}...")
        samples_processed = 0

        for file_path in parquet_files:
            if skip_filename and file_path.name == skip_filename:
                continue

            print(f"Processing {file_path.name}...")
            pf = pq.ParquetFile(file_path)

            for batch in pf.iter_batches(batch_size=batch_size):
                df_batch = pa.Table.from_batches([batch]).to_pandas()

                for idx, row in df_batch.iterrows():
                    if sample_every > 1 and (idx % sample_every) != 0:
                        continue

                    translation = row.get("translation", {})
                    if not isinstance(translation, dict):
                        continue

                    en_text = translation.get("en", "")
                    fr_text = translation.get("fr", "")

                    if en_text and fr_text:
                        self._process_sentence_pair(en_text, fr_text)
                        samples_processed += 1

                        if samples_processed % 10000 == 0:
                            print(f"  Processed {samples_processed} sentence pairs...")

                        if max_samples is not None and samples_processed >= max_samples:
                            print(f"Reached max_samples limit ({max_samples})")
                            break

                if max_samples is not None and samples_processed >= max_samples:
                    break

            if max_samples is not None and samples_processed >= max_samples:
                break

        print("\nCorpus statistics:")
        print(f"  Total sentence pairs processed: {samples_processed}")
        print(f"  Unique target words: {len(self.word_counts)}")
        print(f"  Unique target window pairs: {len(self.pair_counts)}")
        print(f"  Unique target bigrams: {len(self.bigram_counts)}")
        print(f"  Unique source words: {len(self.src_word_counts)}")
        print(f"  Unique src-tgt pairs: {len(self.st_pair_counts)}")

        print("\nComputing target-side PMI scores...")
        self._compute_target_side_pmi()
        print("Target-side PMI computation complete!")

        print("Building src->tgt index...")
        self.build_src_index()
        print("Index ready.")

    def _process_sentence_pair(self, source_text: str, target_text: str) -> None:
        src_words = self._tokenize(source_text)
        tgt_words = self._tokenize(target_text)
        if not src_words or not tgt_words:
            return

        # ---- Target-side counts ----
        for w in tgt_words:
            self.word_counts[w] += 1
            self.total_words += 1

        for i in range(len(tgt_words) - 1):
            self.bigram_counts[f"{tgt_words[i]} {tgt_words[i + 1]}"] += 1

        for i, w1 in enumerate(tgt_words):
            start = max(0, i - self.window_size)
            end = min(len(tgt_words), i + self.window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                w2 = tgt_words[j]
                pair = tuple(sorted((w1, w2)))
                self.pair_counts[pair] += 1
                self.total_pairs += 1

        # ---- Source-side counts (test-time) ----
        for w in src_words:
            self.src_word_counts[w] += 1
            self.total_src_words += 1

        # ---- Source-target joint counts (test-time) ----
        # sentence-pair co-occurrence, no alignments
        for s in src_words:
            for t in tgt_words:
                self.st_pair_counts[(s, t)] += 1
                self.total_st_pairs += 1

    # =========================
    # Target-side PMI (bigram PMI is used to merge unigram picks)
    # =========================
    def _compute_target_side_pmi(self) -> None:
        self.pmi_scores = {}
        self.bigram_pmi_scores = {}

        if self.total_pairs <= 0 or self.total_words <= 0:
            return

        # window PMI
        for (w1, w2), cnt in self.pair_counts.items():
            if cnt < self.min_count:
                continue
            p_xy = cnt / self.total_pairs
            p_x = self.word_counts[w1] / self.total_words
            p_y = self.word_counts[w2] / self.total_words
            if p_x > 0 and p_y > 0:
                self.pmi_scores[(w1, w2)] = math.log(p_xy / (p_x * p_y))

        # consecutive bigram PMI
        denom = max(self.total_words - 1, 1)
        for bigram, cnt in self.bigram_counts.items():
            if cnt < self.min_count:
                continue
            parts = bigram.split()
            if len(parts) != 2:
                continue
            w1, w2 = parts
            p_bigram = cnt / denom
            p_w1 = self.word_counts[w1] / self.total_words
            p_w2 = self.word_counts[w2] / self.total_words
            if p_w1 > 0 and p_w2 > 0:
                self.bigram_pmi_scores[bigram] = math.log(p_bigram / (p_w1 * p_w2))

    # =========================
    # Index building (speed)
    # =========================
    def build_src_index(self) -> None:
        idx: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (src, tgt), cnt in self.st_pair_counts.items():
            idx[src].append((tgt, cnt))
        self.src_to_tgt = idx

    # =========================
    # Test-time constraints: SOURCE only
    # =========================
    def get_constraints_from_source(
        self,
        source_text: str,
        max_constraints: int = 4,
        min_pmi_threshold: float = 0.0,
        min_joint_count: int | None = None,
        min_src_count: int | None = None,
        min_tgt_count: int | None = None,
        prefer_bigrams: bool = False,
        dedup_ratio_thresh: float = 0.88,
        dedup_prefix_len: int = 4,
        pool_multiplier: int = 12,
    ) -> list[str]:
        """
        Select target-side constraints using ONLY the source sentence.

        PMI(s,t) = log( P(s,t) / (P(s)*P(t)) )
        - P(s,t): st_pair_counts / total_st_pairs  (sentence-pair co-occurrence)
        - P(s): src_word_counts / total_src_words
        - P(t): word_counts / total_words

        Then rank candidate target words by their best PMI across any source word in the sentence.
        Finally apply a diversity filter to avoid near-duplicates.

        """
        if min_joint_count is None:
            min_joint_count = self.min_count
        if min_src_count is None:
            min_src_count = self.min_count
        if min_tgt_count is None:
            min_tgt_count = self.min_count

        src_words = self._tokenize(source_text)
        if not src_words:
            return []

        # Need cross-lingual stats
        if self.total_st_pairs <= 0 or self.total_src_words <= 0 or self.total_words <= 0:
            return []

        # Ensure index exists
        if self.src_to_tgt is None:
            self.build_src_index()

        candidates: dict[str, float] = {}

        for s in src_words:
            if self.src_word_counts.get(s, 0) < min_src_count:
                continue

            for tgt, joint_cnt in self.src_to_tgt.get(s, []):  # type: ignore[union-attr]
                if joint_cnt < min_joint_count:
                    continue
                if self.word_counts.get(tgt, 0) < min_tgt_count:
                    continue

                p_st = joint_cnt / self.total_st_pairs
                p_s = self.src_word_counts[s] / self.total_src_words
                p_t = self.word_counts[tgt] / self.total_words
                if p_s <= 0 or p_t <= 0:
                    continue

                pmi = math.log(p_st / (p_s * p_t))
                if pmi >= min_pmi_threshold:
                    prev = candidates.get(tgt)
                    if prev is None or pmi > prev:
                        candidates[tgt] = pmi

        if not candidates:
            return []

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        pool_size = max(max_constraints * pool_multiplier, max_constraints)
        pool = [t for t, _ in ranked[:pool_size]]

        # ---- diversity filter (near-duplicate removal) ----
        selected: list[str] = []
        for cand in pool:
            if any(
                _too_similar(
                    cand,
                    prev,
                    ratio_thresh=dedup_ratio_thresh,
                    prefix_len=dedup_prefix_len,
                )
                for prev in selected
            ):
                continue
            selected.append(cand)
            if len(selected) >= max_constraints:
                break

        if not prefer_bigrams:
            return selected

        # ---- optional bigram preference heuristic ----
        # Merge consecutive unigram picks into a bigram when its target-side PMI is high.
        merged: list[str] = []
        used = set()

        # expand candidate pool for merging
        for i in range(len(selected) - 1):
            w1, w2 = selected[i], selected[i + 1]
            if w1 in used or w2 in used:
                continue
            bigram = f"{w1} {w2}"
            if self.bigram_pmi_scores.get(bigram, float("-inf")) >= min_pmi_threshold:
                merged.append(bigram)
                used.add(w1)
                used.add(w2)
            if len(merged) >= max_constraints:
                break

        # fill with remaining unigrams
        for w in selected:
            if len(merged) >= max_constraints:
                break
            if w not in used:
                merged.append(w)

        # final diversity filter (handles cases like "word" + "word word")
        final: list[str] = []
        for cand in merged:
            if any(
                _too_similar(
                    cand,
                    prev,
                    ratio_thresh=dedup_ratio_thresh,
                    prefix_len=dedup_prefix_len,
                )
                for prev in final
            ):
                continue
            final.append(cand)
            if len(final) >= max_constraints:
                break

        return final

    # =========================
    # Save/load
    # =========================
    def save_statistics(self, output_path: str) -> None:
        import pickle

        stats = {
            # target-side
            "word_counts": dict(self.word_counts),
            "pair_counts": dict(self.pair_counts),
            "bigram_counts": dict(self.bigram_counts),
            "pmi_scores": self.pmi_scores,
            "bigram_pmi_scores": self.bigram_pmi_scores,
            "total_words": self.total_words,
            "total_pairs": self.total_pairs,
            # source-side + cross-lingual
            "src_word_counts": dict(self.src_word_counts),
            "st_pair_counts": dict(self.st_pair_counts),
            "total_src_words": self.total_src_words,
            "total_st_pairs": self.total_st_pairs,
            # config
            "min_count": self.min_count,
            "window_size": self.window_size,
        }

        with open(output_path, "wb") as f:
            pickle.dump(stats, f)

        print(f"Statistics saved to {output_path}")

    def load_statistics(self, input_path: str) -> None:
        import pickle

        with open(input_path, "rb") as f:
            stats = pickle.load(f)

        # target-side
        self.word_counts = Counter(stats.get("word_counts", {}))
        self.pair_counts = defaultdict(int, stats.get("pair_counts", {}))
        self.bigram_counts = Counter(stats.get("bigram_counts", {}))
        self.pmi_scores = stats.get("pmi_scores", {})
        self.bigram_pmi_scores = stats.get("bigram_pmi_scores", {})
        self.total_words = int(stats.get("total_words", 0))
        self.total_pairs = int(stats.get("total_pairs", 0))

        # source-side + cross-lingual (may be missing in legacy pickles)
        self.src_word_counts = Counter(stats.get("src_word_counts", {}))
        self.st_pair_counts = defaultdict(int, stats.get("st_pair_counts", {}))
        self.total_src_words = int(stats.get("total_src_words", 0))
        self.total_st_pairs = int(stats.get("total_st_pairs", 0))

        # config
        self.min_count = int(stats.get("min_count", self.min_count))
        self.window_size = int(stats.get("window_size", self.window_size))

        # build index if possible
        if len(self.st_pair_counts) > 0:
            self.build_src_index()
        else:
            self.src_to_tgt = None
            print(
                "⚠️ Loaded legacy stats without src↔tgt counts. "
                "To use get_constraints_from_source(), rebuild and resave stats once."
            )

        print(f"Statistics loaded from {input_path}")
        print(f"  Unique target words: {len(self.word_counts)}")
        print(f"  PMI scores computed: {len(self.pmi_scores)}")
        print(f"  Unique source words: {len(self.src_word_counts)}")
        print(f"  Unique src-tgt pairs: {len(self.st_pair_counts)}")
