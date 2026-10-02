"""Spike: can DiffusionGemma's bidirectional canvas find a seeded bug?

Self-contained -- it does not import ``softmaxlint`` and does not touch ``src/``.
Its only job is to answer one question before anyone rewrites the linter:

    given a buggy snippet and the corrected one, does the model's per-token
    distribution actually separate them?

If the buggy `<=` scores far below the corrected `<`, the approach works. If
both look alike, we learn that cheaply instead of after a migration.

How it works
------------
DiffusionGemma is an encoder-decoder. The encoder runs *causally* and caches
K/V; the decoder applies *bidirectional* attention across a canvas while
cross-attending to that cache. So one forward pass can score a token given the
code on **both** sides of it::

    context (encoder) = prefix ++ suffix      concatenated; order is irrelevant
    canvas  (decoder) = the line, with position k replaced by uniform noise

Position k is the only masked position, and every other token of the line is
visible to it through bidirectional self-attention. That is exactly Fill-In-The-
Middle scoring -- structural, rather than a prompt convention the model might
not have been trained on.

It also returns logits over the full 262K vocabulary, so there is no top-k
censoring: every token gets an exact probability.

Run it::

    uv run --extra spike python scripts/spike_diffusion_scorer.py --awq --vram-gib 18

Verified so far, on ``trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion``
(8.2 MB, same architecture, random weights):

* ``logits`` comes back as ``(1, canvas_length, 262144)`` -- the **full**
  vocabulary, so unlike the OpenAI completions API there is no top-k censoring
  and every token gets an exact probability.
* Masking one canvas position changes that position's distribution, so the
  decoder really is reading the canvas.
* Canvas positions are *not* conditionally independent: perturbing position 3
  also moves positions 0-2 slightly, because canvas attention is bidirectional.
  Scoring still works -- mask position k, read position k, discard the rest --
  but it does mean the canvas cannot be scored as if its positions were
  independent.

Not yet verified: whether the real 26B model separates a buggy ``<=`` from a
correct ``<``. See the VRAM note in :func:`load`.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from transformers import DiffusionGemmaForBlockDiffusion
    from transformers.tokenization_utils_base import BatchEncoding

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

#: Pre-quantized to 4-bit. On-the-fly bitsandbytes quantization does not work
#: for this model (see `load`), and the bf16 weights are 48 GB.
AWQ_MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"

BUGGY_LINE = "        if i <= len(numbers):\n"
FIXED_LINE = "        if i < len(numbers):\n"

BUGGY_CONTEXT = "def average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n"
SUFFIX = "            total += numbers[i]\n    return total / len(numbers)\n"

SEPARATION_TARGET = 4.0
"""How many times more surprising the buggy token must be to call this a pass."""


class TokenScore(TypedDict):
    """One scored token: what was written against what the model wanted."""

    text: str
    actual_prob: float
    top_prob: float
    top_text: str
    ratio: float


@runtime_checkable
class SupportsEncodeDecode(Protocol):
    """The slice of the tokenizer API this spike uses.

    ``AutoTokenizer`` is a factory class in transformers 5, so its instance
    methods are invisible to a type checker. Stating the two we call keeps the
    signature honest instead of reaching for ``Any``.
    """

    def __call__(self, text: str, *, return_tensors: str, add_special_tokens: bool) -> BatchEncoding: ...

    def decode(self, token_ids: list[int]) -> str: ...


def load(
    model_id: str,
    *,
    tokenizer_id: str,
    quantize: bool,
    vram_gib: int,
) -> tuple[DiffusionGemmaForBlockDiffusion, SupportsEncodeDecode]:
    """Return (model, tokenizer), in 4-bit if asked."""
    import torch
    from transformers import AutoConfig, AutoTokenizer, DiffusionGemmaForBlockDiffusion
    from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsConfig

    print(f"loading {model_id} ({'4-bit nf4' if quantize else 'bf16'}) ...", flush=True)

    if quantize:
        # 88% of this model is MoE expert stacks (42.5 of 48.1 GiB), and
        # bitsandbytes does not quantize 3-D expert tensors, so they stay bf16
        # and the model wants ~46 GiB regardless of load_in_4bit. That is why a
        # 24 GB card OOMs on a single 22 GiB allocation, and why the offload
        # path dies on a meta tensor.
        #
        # --awq loads a pre-quantized checkpoint instead. Transformers insists
        # the config class match the one the checkpoint was saved with, hence
        # reading it from the checkpoint itself.
        config = AutoConfig.from_pretrained(model_id)
        quant_config = CompressedTensorsConfig(**dict(config.quantization_config))
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_id,
            config=config,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: f"{vram_gib}GiB"},
            quantization_config=quant_config,
        )
    else:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer is None or not isinstance(tokenizer, SupportsEncodeDecode):
        # AutoTokenizer resolves to one of several backends; only the ones with
        # the encode/decode surface this spike uses can score anything.
        msg = f"tokenizer for {tokenizer_id} lacks the encode/decode API this spike needs"
        raise ValueError(msg)
    return model, tokenizer


def score_line(
    model: DiffusionGemmaForBlockDiffusion,
    tokenizer: SupportsEncodeDecode,
    line: str,
    *,
    seed: int,
) -> list[TokenScore]:
    """Score every token of ``line`` in turn, each masked alone.

    The context is the whole rest of the function: the code above as the prefix
    and the code below as the suffix, so each token is judged with the code that
    follows it as well as the code that came before.
    """
    import torch

    device = next(model.parameters()).device
    text_config = model.config.text_config
    if text_config is None or isinstance(text_config, dict):
        msg = "model config is missing text_config.vocab_size"
        raise ValueError(msg)
    vocab_size = int(text_config.vocab_size)

    context = tokenizer(BUGGY_CONTEXT + SUFFIX, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    canvas = tokenizer(line, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    attention = torch.ones_like(context)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    rows: list[TokenScore] = []
    for position, token_id in enumerate(canvas[0].tolist()):
        # Mask exactly one position with uniform noise; every other token of the
        # line stays visible, so the model reads it rather than copying it.
        noised = canvas.clone()
        noised[0, position] = torch.randint(low=0, high=vocab_size, size=(1,), generator=generator).to(device)

        with torch.no_grad():
            out = model(
                input_ids=context,
                attention_mask=attention,
                decoder_input_ids=noised,
                decoder_attention_mask=None,  # built internally: bidirectional over canvas + cache
                self_conditioning_logits=None,  # first denoising step
            )

        probs = out.logits[0, position].float().softmax(dim=-1)
        actual = probs[token_id].item()
        top_prob, top_id = probs.max(dim=-1)
        top_prob = top_prob.item()

        rows.append(
            {
                "text": tokenizer.decode([token_id]),
                "actual_prob": actual,
                "top_prob": top_prob,
                "top_text": tokenizer.decode([top_id]),
                "ratio": actual / top_prob if top_prob > 0 else 1.0,
            }
        )
    return rows


def show(label: str, rows: list[TokenScore], suspect: str) -> float | None:
    print(f"\n=== {label} ===")
    print(f"{'token':<14} {'top':<14} {'ratio':>10} {'act%':>8}")
    for row in rows:
        mark = "  <<< suspect" if row["text"] == suspect else ""
        print(
            f"{row['text']!r:<14} {row['top_text']!r:<14} {row['ratio']:>10.4f} {row['actual_prob'] * 100:>7.2f}%{mark}"
        )

    ratios = [row["ratio"] for row in rows]
    print(f"\nmin ratio {min(ratios):.4f}   median {sorted(ratios)[len(ratios) // 2]:.4f}")
    return next((row["ratio"] for row in rows if row["text"] == suspect), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--awq", action="store_true", help="use the pre-quantized 4-bit checkpoint")
    parser.add_argument("--tokenizer", default=MODEL_ID, help="tokenizer to load (the AWQ repo may omit it)")
    parser.add_argument("--no-quantize", action="store_true", help="load in bf16 (needs ~48 GB)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vram-gib", type=int, default=20, help="VRAM budget for 4-bit placement")
    args = parser.parse_args()

    model_id = AWQ_MODEL_ID if args.awq else args.model
    quantize = args.awq or not args.no_quantize

    try:
        model, tokenizer = load(model_id, tokenizer_id=args.tokenizer, quantize=quantize, vram_gib=args.vram_gib)
    except ValueError:
        # Quantized repos often ship weights only, so fall back to the official
        # tokenizer rather than making the caller know that.
        print("retrying with the official tokenizer ...", flush=True)
        try:
            model, tokenizer = load(model_id, tokenizer_id=MODEL_ID, quantize=quantize, vram_gib=args.vram_gib)
        except Exception as exc:  # noqa: BLE001
            print(f"\ncould not load the model: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("\nThis spike needs the 'spike' extra and roughly 14 GB of free VRAM.", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\ncould not load the model: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("\nThis spike needs the 'spike' extra and roughly 14 GB of free VRAM.", file=sys.stderr)
        return 2

    buggy = show("BUGGY:  `if i <= len(numbers):`", score_line(model, tokenizer, BUGGY_LINE, seed=args.seed), "<=")
    fixed = show("FIXED:  `if i < len(numbers):`", score_line(model, tokenizer, FIXED_LINE, seed=args.seed), "<")

    print("\n=== VERDICT ===")
    if not buggy or not fixed:
        print("could not isolate the boundary operator; the canvas did not contain it as its own token")
        return 1
    separation = fixed / buggy if buggy else float("inf")
    print(f"buggy '<=' ratio {buggy:.6f}   fixed '<' ratio {fixed:.6f}   separation {separation:.2f}x")
    if separation >= SEPARATION_TARGET:
        print(f"\nPASS: separated by {separation:.2f}x (target {SEPARATION_TARGET}x).")
        print("Bidirectional canvas scoring works; a real migration is justified.")
        return 0
    print(f"\nFAIL: separation {separation:.2f}x is below the {SEPARATION_TARGET}x target.")
    print("Either 4-bit quantization blunted the signal, or this metric cannot see this bug.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
