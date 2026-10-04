# softmaxlint

Probabilistic Python linter. It scores each token against a local vLLM server
with Fill-In-The-Middle prompts and flags tokens the model found improbable,
rendered as a Rich heatmap. Exits 0 when clean, 1 on findings, 2 on hard error.

```sh
just run -- check path/to/code.py
```

## Serving a model

The linter needs a FIM-trained model on `http://127.0.0.1:18020/v1`
(`SERVER_BASE_URL`):

```sh
just up
MODEL_NAME=coder-7b just calibrate
just down
```

The stack serves Qwen2.5-Coder-7B. Qwen3-family checkpoints declare
`<|fim_*|>` tokens but were not trained for FIM and answer every hole with a
newline, so they produce no signal. Measured against `coder-7b`: FP ~1.2% of
tokens, recall 3/4 seeded bug types (`corpus/`).

## Development

```sh
just check   # ruff + ty
just test    # pytest (stubbed server, no GPU needed)
just ci      # what CI runs
```

Live runs need the server above; scoring a file takes a few dozen server
round-trips, so large trees take a while. `scripts/calibrate.py` reports false-positive rate, recall, and
censoring rate over `corpus/`; run it before trusting a change to scoring or
grading.
