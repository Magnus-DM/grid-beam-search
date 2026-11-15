"""Grid beam search (Hokamp and Liu, ACL 2017) for lexically constrained decoding.

Hypotheses are kept in a grid indexed by c, the number of constraint tokens covered
so far. Each step applies three operations:

- GENERATE: an open hypothesis adds one of its top-k free tokens (c stays the same);
- START: an open hypothesis forces the first token of an unmet constraint (c + 1);
- CONTINUE: a hypothesis with a constraint in progress forces its next token (c + 1).

At most one constraint is in progress at a time, hypotheses that can no longer fit
the remaining constraint tokens before max_length are pruned, and EOS is only
allowed once every constraint is satisfied. Each grid cell keeps its beam_width best
hypotheses by length-normalised log-probability.
"""

from collections import defaultdict
from copy import deepcopy

import torch
import tqdm

from gbs.baseline import GeneratedSequence, ScoredToken

UNK_TOKEN_ID = 2  # T5 <unk>


class ConstrainedSequence(GeneratedSequence):
    """A hypothesis that also tracks progress on each constraint.

    constraints: one token-id list per constraint (a constraint can span several tokens).
    match_positions[i]: how many tokens of constraint i are matched so far.
    satisfied[i]: whether constraint i has been produced completely.
    """

    def __init__(
        self,
        initial_token: int,
        end_token_id: int,
        initial_score: float,
        constraints: list[list[int]],
    ):
        super().__init__(initial_token, end_token_id, initial_score)
        self.constraints: list[list[int]] = constraints
        self.match_positions: list[int] = [0 for _ in constraints]
        self.satisfied: list[bool] = [False for _ in constraints]

    def __deepcopy__(self, memo):
        # Constraints are read-only, so copies share them; the token list is copied shallowly.
        new_obj = ConstrainedSequence(None, self.end_token_id, 0.0, self.constraints)
        new_obj.sequence = self.sequence.copy()
        new_obj._score = self._score
        new_obj.normalized_score = self.normalized_score
        new_obj.match_positions = self.match_positions[:]
        new_obj.satisfied = self.satisfied[:]
        new_obj._ids_cache = None
        return new_obj

    def append(self, scored_token: ScoredToken, action: str = "start"):
        super().append(scored_token)
        self._update_constraints_with_token(scored_token.token_id, action)

    def _update_constraints_with_token(self, token_id: int, action: str = "start"):
        for i, cons in enumerate(self.constraints):
            if self.satisfied[i] or len(cons) == 0:
                continue
            pos = self.match_positions[i]
            # A CONTINUE step only advances the constraint that is already in progress.
            if pos == 0 and action == "continue":
                continue
            if token_id == cons[pos]:
                pos += 1
            else:
                # Restart the match if this token is the constraint's first token.
                pos = 1 if token_id == cons[0] else 0
            if pos >= len(cons):
                self.satisfied[i] = True
                pos = 0
            self.match_positions[i] = pos

    def satisfied_count(self) -> int:
        return sum(1 for x in self.satisfied if x)

    def all_satisfied(self) -> bool:
        return all(self.satisfied) if self.constraints else True

    def in_progress_indices(self) -> list[int]:
        """Constraints that are partly matched but not yet satisfied."""
        return [
            i
            for i, (pos, sat) in enumerate(zip(self.match_positions, self.satisfied))
            if (pos > 0 and not sat)
        ]

    def unmet_indices(self) -> list[int]:
        return [i for i, sat in enumerate(self.satisfied) if not sat]

    def next_required_tokens(self) -> list[int]:
        """For each in-progress constraint, the token that must come next."""
        required = []
        for i in self.in_progress_indices():
            pos = self.match_positions[i]
            cons = self.constraints[i]
            if pos < len(cons):
                required.append(cons[pos])
        return required

    def covered_constraint_tokens(self) -> int:
        """Number of constraint tokens covered so far: the grid coordinate c."""
        total = 0
        for cons, sat, pos in zip(self.constraints, self.satisfied, self.match_positions):
            total += len(cons) if sat else pos
        return total


def _ensure_constraint_ids(
    tokenizer,
    constraints: list[str] | list[list[int]],
) -> list[list[int]]:
    """Accept constraints as strings or token-id lists; return token-id lists."""
    if not constraints:
        return []
    if isinstance(constraints[0], list):
        return constraints  # type: ignore[return-value]
    return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in constraints]


def grid_beam_search(
    tokenizer,
    model,
    src_input_ids: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    device: torch.device,
    constraints: list[str] | list[list[int]],
    beam_width: int = 3,
    temperature: float = 1.0,
    max_length: int = 64,
    verbose: bool = True,
):
    """Decode a translation that contains every constraint.

    Returns up to beam_width hypotheses, best first. Hypotheses that ended with EOS
    and satisfy all constraints are preferred; otherwise the best hypotheses that
    satisfy all constraints; otherwise those that cover the most constraint tokens.
    """
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids=src_input_ids, return_dict=True)

    constraint_ids = _ensure_constraint_ids(tokenizer, constraints)
    num_c = sum(len(cons) for cons in constraint_ids)  # total constraint tokens

    if verbose:
        print("\nConstraint tokenization:")
        for orig, ids in zip(constraints, constraint_ids):
            print(f"  {orig!r} -> {ids}")
        print(f"Total constraint tokens: {num_c}")

    previous_step = defaultdict(list)  # c -> hypotheses at time t
    next_step = defaultdict(list)  # c -> hypotheses at time t + 1

    previous_step[0].append(
        ConstrainedSequence(
            initial_token=start_token_id,
            end_token_id=end_token_id,
            initial_score=0.0,
            constraints=constraint_ids,
        )
    )
    finished_hypotheses: list[ConstrainedSequence] = []

    def add_if_reachable(new_hyp, t):
        new_c = min(new_hyp.covered_constraint_tokens(), num_c)
        if num_c - new_c <= max_length - (t + 1):
            next_step[new_c].append(new_hyp)

    for t in tqdm.tqdm(range(1, max_length), desc="Grid beam search", disable=not verbose):
        next_step.clear()
        remaining_steps = max_length - t

        for c, hyps in list(previous_step.items()):
            # Not enough steps left to cover the remaining constraint tokens.
            if num_c - c > remaining_steps:
                continue

            for hyp in hyps:
                if hyp.has_ended():
                    if hyp.all_satisfied() and hyp.covered_constraint_tokens() == num_c:
                        finished_hypotheses.append(hyp)
                    continue

                gen_seq = torch.tensor(hyp.ids(), device=device, dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    output = model(encoder_outputs=encoder_outputs, decoder_input_ids=gen_seq)
                    logits = output.logits[:, -1, :].squeeze(0)

                scaled_logits = logits / max(temperature, 1e-8)
                scaled_logits[UNK_TOKEN_ID] = float("-inf")
                # EOS is only allowed once all constraints are covered.
                if not (hyp.all_satisfied() and hyp.covered_constraint_tokens() == num_c):
                    scaled_logits[end_token_id] = float("-inf")
                probs = torch.softmax(scaled_logits, dim=-1)

                if hyp.in_progress_indices():
                    # CONTINUE: a constraint is in progress, so its next token is forced.
                    for tok_id in hyp.next_required_tokens():
                        new_hyp = deepcopy(hyp)
                        new_hyp.append(
                            ScoredToken(tok_id, torch.log(probs[tok_id]).item()),
                            action="continue",
                        )
                        add_if_reachable(new_hyp, t)
                else:
                    # GENERATE: extend with the top-k free tokens.
                    topk_probs, topk_ids = torch.topk(probs, beam_width)
                    for token_id, prob in zip(topk_ids.tolist(), topk_probs.tolist()):
                        log_prob = float(torch.log(torch.tensor(prob + 1e-30)))
                        new_hyp = deepcopy(hyp)
                        new_hyp.append(ScoredToken(token_id, log_prob))
                        add_if_reachable(new_hyp, t)

                    # START: force the first token of each unmet constraint.
                    for cons_idx in hyp.unmet_indices():
                        cons = constraint_ids[cons_idx]
                        if not cons:
                            continue
                        first_tok = cons[0]
                        new_hyp = deepcopy(hyp)
                        new_hyp.append(ScoredToken(first_tok, torch.log(probs[first_tok]).item()))
                        add_if_reachable(new_hyp, t)

        # Keep the beam_width best hypotheses in every grid cell.
        pruned_next = defaultdict(list)
        for c, hyps in next_step.items():
            if hyps:
                pruned_next[c] = sorted(hyps, key=lambda h: h.normalized_score, reverse=True)[
                    :beam_width
                ]
        previous_step = pruned_next

    if finished_hypotheses:
        finished_hypotheses.sort(key=lambda h: h.normalized_score, reverse=True)
        return finished_hypotheses[:beam_width]

    # No hypothesis reached EOS: prefer those that satisfy all constraints.
    all_grid = [h for hyps in previous_step.values() for h in hyps]
    satisfied = [h for h in all_grid if h.all_satisfied()]
    if satisfied:
        satisfied.sort(key=lambda h: h.normalized_score, reverse=True)
        return satisfied[:beam_width]

    all_grid.sort(key=lambda h: (h.covered_constraint_tokens(), h.normalized_score), reverse=True)
    return all_grid[:beam_width]
