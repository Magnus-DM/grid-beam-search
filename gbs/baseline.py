"""Baseline beam search and evaluation.

Based on the decoding exercise template of the Advanced Natural Language Processing
course (KU Leuven, 2025): `ScoredToken`, `GeneratedSequence`, `beam_search` and
`evaluate` follow that template. My changes are performance only: cached encoder
outputs, a cached id list and tensors created directly on the target device.
This module is the baseline that grid beam search is compared against.
"""

import dataclasses
from copy import deepcopy

import sacrebleu
import torch
import tqdm


@dataclasses.dataclass
class ScoredToken:
    """A token and its log-probability."""

    token_id: int
    score: float


class GeneratedSequence:
    """A sequence in the process of being generated, with its cumulative score."""

    def __init__(self, initial_token, end_token_id, initial_score):
        self.end_token_id = end_token_id
        self._score = initial_score  # cumulative log-probability
        self.normalized_score = initial_score
        self.sequence = [ScoredToken(initial_token, initial_score)]
        self._ids_cache = None

    def append(self, scored_token):
        """Append a token and update the raw and length-normalised scores."""
        self.sequence.append(scored_token)
        self._score += scored_token.score
        self.normalized_score = self._score / len(self.sequence)
        self._ids_cache = None

    def ids(self):
        if self._ids_cache is None:
            self._ids_cache = [st.token_id for st in self.sequence]
        return self._ids_cache

    def has_ended(self):
        """True if the last token is the end-of-sequence token."""
        return self.sequence and self.sequence[-1].token_id == self.end_token_id

    def __lt__(self, other):
        # Needed to sort candidates by score in beam_search.
        return self.normalized_score < other.normalized_score

    def __str__(self):
        return f"{self._score: .6f}({self.normalized_score: .6f}): {self.sequence}"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        gs = GeneratedSequence(None, self.end_token_id, 0.0)
        gs.sequence = self.sequence.copy()
        gs._score = self._score
        gs.normalized_score = self.normalized_score
        gs._ids_cache = None
        return gs


def detect_device():
    """Use the GPU when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search(
    tokenizer,
    model,
    src_input_ids: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    device: torch.device,
    beam_width: int = 8,
    temperature: float = 1.0,
    max_length: int = 64,
    verbose: bool = True,
):
    """Standard beam search with length-normalised scores.

    Returns the final candidate sequences, best first.
    """
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids=src_input_ids, return_dict=True)

    candidate_sequences = [GeneratedSequence(start_token_id, end_token_id, 0.0)]

    for _ in tqdm.tqdm(range(max_length - 1), desc="Beam search", disable=not verbose):
        next_step_candidates = []

        for candidate in candidate_sequences:
            # Finished sequences are carried over unchanged.
            if candidate.has_ended():
                next_step_candidates.append(candidate)
                continue

            gen_seq = torch.tensor(candidate.ids(), device=device, dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                output = model(encoder_outputs=encoder_outputs, decoder_input_ids=gen_seq)

            logits = output.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(beam_width)

            if verbose:
                print(
                    f"{candidate.normalized_score}: [{candidate.ids()}], next token probabilities:"
                )
                for p, w in zip(
                    top_probs.tolist()[0],
                    tokenizer.convert_ids_to_tokens(top_ids.tolist()[0]),
                ):
                    print(f"\tp: {p: .6f}: {w}")

            for i in range(beam_width):
                new_seq = deepcopy(candidate)
                new_seq.append(ScoredToken(top_ids[:, i].item(), torch.log(top_probs[:, i]).item()))
                next_step_candidates.append(new_seq)

        # Keep the beam_width best candidates.
        next_step_candidates.sort(reverse=True)
        candidate_sequences = next_step_candidates[:beam_width]

        if all(seq.has_ended() for seq in candidate_sequences):
            break

    return candidate_sequences


@dataclasses.dataclass
class EvaluationResult:
    bleu: float
    chrf: float
    chrf_pp: float


def evaluate(hypotheses, references):
    """Corpus-level BLEU, chrF and chrF++ against a single reference per sentence."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    chrf_pp = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,
        char_order=6,
        beta=2,
        eps_smoothing=False,
    ).score
    return EvaluationResult(bleu, chrf, chrf_pp)
