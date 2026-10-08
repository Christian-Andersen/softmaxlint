# softmaxlint

A linter with no rules: every token is scored in place by a code model with
the whole file as context, and tokens the model finds improbable are
flagged. Served over a local vLLM endpoint, reported as a heatmap, editor
diagnostics, or concise lines. Exits 0 when clean, 1 on findings, 2 on hard
error.

## Example

```python
"""Seeded defects: wrong operators and wrong operands."""


def is_empty(items):
    # BUG: "==" only matches by value; identity is what was written.
    return items == []


def all_positive(numbers):
    # BUG: "<" should be ">="; zero is not positive but this accepts it.
    return all(number < 0 for number in numbers)


def grow(items):
    # BUG: "-" subtracts where "+" is meant.
    return items - 1
```

```sh
$ softmaxlint lsp check --output-format concise operators.py
operators.py:6:17: [CRITICAL] Wrote ' ==' (0.10%) but the model expected ' is' (94.65%)
operators.py:16:17: [CRITICAL] Wrote ' -' (0.24%) but the model expected ' +' (67.36%)
```

No rule names `==`-on-empty-list. The model read the function, decided the
token was wrong at 0.10%, and named the fix at 94.65%. The middle function
is not flagged: whether zero counts as positive lives in the function's
spec, which the model can't see. Surprisal catches wrong tokens, not wrong
requirements — the Calibration numbers below say the same.

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
silent, `corpus/bugs` carries seeded defects). Against Qwen2.5-Coder-7B:

- False positives: ~1.2% of tokens on known-good code
- Recall: 3/4 seeded bug types, with all 4 buggy files flagged
- The miss is off-by-one boundaries (`<=` vs `<`), where the model is
  genuinely unsure — surprisal cannot see what needs execution to decide

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
