"""Random constraints taken from the reference translation (oracle setting)."""

import random
import re
import zlib

WORD_PATTERN = r"\b\w+'\w+\b|\b\w+’\w+\b|\b\w+\b"


def extract_random_constraints(fr_text, max_num_constraints=4, seed=None, bigram_prob=0.5):
    """Pick 1 to max_num_constraints words or word pairs from a French reference.

    At most floor(num_words / 2) constraints are chosen, so a 4-word sentence gets at
    most 2 and a 1-word sentence none. Each constraint is a bigram with probability
    bigram_prob, otherwise a single word; constraints never overlap.

    The random state is seeded with seed + a CRC32 of the sentence, so the choice is
    the same in every run and on every machine.
    """
    rng = random.Random(seed + zlib.crc32(fr_text.encode("utf-8"))) if seed is not None else random

    words = re.findall(WORD_PATTERN, fr_text)
    max_allowed = len(words) // 2
    if max_allowed == 0:
        return []

    num_constraints = rng.randint(1, min(max_num_constraints, max_allowed))
    selected_constraints = []
    used_positions = set()

    for _ in range(num_constraints):
        valid_positions = [p for p in range(len(words)) if p not in used_positions]
        if not valid_positions:
            break

        is_bigram = rng.random() < bigram_prob
        bigram_positions = [
            p for p in valid_positions if p < len(words) - 1 and (p + 1) not in used_positions
        ]
        if is_bigram and bigram_positions:
            pos = rng.choice(bigram_positions)
            selected_constraints.append(f"{words[pos]} {words[pos + 1]}")
            used_positions.update((pos, pos + 1))
        else:
            pos = rng.choice(valid_positions)
            selected_constraints.append(words[pos])
            used_positions.add(pos)

    return selected_constraints
