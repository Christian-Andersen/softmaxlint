set dotenv-load

alias c := check
alias r := run
alias t := test

[private]
default:
    @just --list

check:
    uv run -- prek run --all-files

audit:
    uv run -- prek run audit --all-files

test:
    uv run -- prek run test --all-files --verbose

run *args:
    uv run -- softmaxlint {{ args }}

calibrate *args:
    uv run -- python scripts/calibrate.py {{ args }}

up:
    podman-compose up -d

down:
    podman-compose down

logs:
    podman-compose logs -f

ci:
    uv run -- prek run --stage manual --all-files
