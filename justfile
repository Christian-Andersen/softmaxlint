set dotenv-load

alias c := check
alias r := run
alias t := test
alias ci := continuous_integration

[private]
default:
    @just --list

check:
    uv run -- prek run --all-files

manual:
    uv run -- prek run --stage manual --all-files

audit:
    uv run -- prek run audit --all-files

test:
    uv run -- prek run test --all-files

run *args:
    uv run -- softmaxlint {{ args }}

continuous_integration: check manual
