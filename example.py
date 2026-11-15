"""Translate one sentence with three lexical constraints (about 30 seconds on a laptop CPU)."""

from transformers import T5ForConditionalGeneration, T5Tokenizer

from gbs.baseline import beam_search, detect_device
from gbs.grid_beam_search import grid_beam_search

PROMPT = "translate English to French: "
SOURCE = "Solve the riddle lurking behind the murder."
CONSTRAINTS = ["Résoudre", "se cache", "mystère"]


def main():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    device = detect_device()
    model.to(device).eval()

    input_ids = tokenizer.encode(PROMPT + SOURCE, return_tensors="pt").to(device)
    common = dict(
        tokenizer=tokenizer,
        model=model,
        src_input_ids=input_ids,
        start_token_id=tokenizer.pad_token_id,
        end_token_id=tokenizer.eos_token_id,
        device=device,
        beam_width=3,
        max_length=32,
        verbose=False,
    )

    print(f"Source:      {SOURCE}")
    print(f"Constraints: {CONSTRAINTS}")
    beam = beam_search(**common)[0]
    print(f"Beam search: {tokenizer.decode(beam.ids(), skip_special_tokens=True)}")
    for seq in grid_beam_search(constraints=CONSTRAINTS, **common):
        text = tokenizer.decode(seq.ids(), skip_special_tokens=True)
        print(f"Grid beam:   {text}  (score {seq.normalized_score:.3f})")


if __name__ == "__main__":
    main()
