# softmaxlint

A linter with no rules: every token is scored in place by a code model with
the whole file as context, and tokens the model finds improbable are
flagged. Served over a local vLLM endpoint, reported as a heatmap, editor
diagnostics, or concise lines. Exits 0 when clean, 1 on findings, 2 on hard
error.

## Example

`control_flow.py:1-16`:

```python
"""Iteration helpers."""


def find_first(items, predicate):
    for item in items:
        if item:
            return item
    return None


def only_even(numbers):
    result = []
    for number in numbers:
        continue
        result.append(number)
    return result
```

```sh
$ softmaxlint lsp check --output-format concise control_flow.py
control_flow.py:6:11: [CRITICAL] Wrote ' item' (0.01%) but the model expected ' predicate' (96.73%)
control_flow.py:14:8: [CRITICAL] Wrote ' continue' (0.01%) but the model expected ' if' (99.11%)
```

No rule names "unconditional `continue`". The model read the loop, decided
the token was wrong at 0.01%, and named the fix at 99.11%.

## How it works

- The server tokenizes each file (echo probe), then every model token gets a
  Fill-In-The-Middle prompt: the file before it as prefix, the file after it
  as suffix, scored in batches of 256 in one request per batch.
- The grade is a likelihood ratio (written vs the model's top pick), gated on
  the model being peaked: flat distributions, whitespace, strings, comments,
  and docstring-or-indent suggestions never flag.
- Only a FIM-trained model produces a signal. Qwen3-family checkpoints answer
  every hole with a newline; Qwen2.5-Coder fills holes as expected.

## Calibration

`scripts/calibrate.py` scores a labelled corpus (`corpus/correct` should stay
silent; `corpus/bugs` lists its seeded defects in `expected.json` beside
them, never in the sources where the model could read them). Against
Qwen2.5-Coder-7B:

- False positives: ~1.2% of tokens on known-good code (10/820)
- Recall: 8/15 seeded bugs, with all 4 buggy files flagged
- The misses cluster where the model is unsure or agrees: boundary `<=`
  with a flat distribution, `==` with no hinting comment nearby, `is`
  against a default and `>=` on exact fits that the model accepts

Labels stay out of model context on purpose: an earlier corpus named the
fix in a comment next to the bug, and recall measured that setup instead
of the tool.

## Serving a model

The linter needs the model on `http://127.0.0.1:18020/v1` (`SERVER_BASE_URL`):

```sh
just up
MODEL_NAME=coder-7b just calibrate
just down
```

The stack serves Qwen2.5-Coder-7B (`compose.yaml`).

## Development

```sh
just check   # ruff + ty
just test    # pytest (stubbed server, no GPU needed)
just ci      # what CI runs
```

Live runs need the server above; scoring a file takes a few dozen server
round-trips, so large trees take a while. Run `just calibrate` before
trusting a change to scoring or grading.

## Editor integration (LSP)

`softmaxlint lsp server` speaks the Language Server Protocol over stdio, so
any editor with an LSP client works. It lints a file on open and on save and
publishes the findings as diagnostics. Neovim:

```lua
vim.api.nvim_create_autocmd('FileType', {
  pattern = 'python',
  callback = function()
    vim.lsp.start({
      name = 'softmaxlint',
      cmd = { 'softmaxlint', 'lsp', 'server' },
      root_dir = vim.fs.root(0, { '.git' }),
    })
  end,
})
```

Without a running LSP client, `softmaxlint lsp check <path>` runs the same
scoring and grading but prints machine-readable output: LSP diagnostics JSON
by default, or ruff-style lines with `--output-format concise` (also
`OUTPUT_FORMAT`).
