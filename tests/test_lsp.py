"""LSP diagnostics mapping, one-shot JSON output, and the stdio server loop."""

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from typer.testing import CliRunner

from softmaxlint import cli, lsp
from softmaxlint.linter import SoftmaxLinter
from softmaxlint.settings import Settings
from tests.conftest import (
    BUG_DIR,
    BUGGY_OPERATOR,
    BUGGY_SNIPPET,
    StubServer,
    fim_responder,
)

runner = CliRunner()

MakeLinter = Callable[..., SoftmaxLinter]
Serve = Callable[..., None]


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch, settings: Settings, server: StubServer) -> Serve:
    """Point the CLI at a stub vLLM endpoint, mirroring test_cli."""

    def _serve(responder: Callable[[str | list[str]], dict[str, Any]] | None = None) -> None:
        server.responder = responder

        class _StubbedLinter(SoftmaxLinter):
            def __init__(self, config: Settings | None = None) -> None:
                super().__init__(config)
                self.client = server.client()

        monkeypatch.setattr(cli, "Settings", lambda: settings)
        monkeypatch.setattr(cli, "SoftmaxLinter", _StubbedLinter)

    return _serve


def frame(payload: dict[str, Any]) -> bytes:
    """One Content-Length framed message, as the server reads them."""
    body = json.dumps(payload).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def outgoing(buffer: io.BytesIO) -> list[dict[str, Any]]:
    """Every message the server wrote, in order."""
    buffer.seek(0)
    messages = []
    while True:
        line = buffer.readline()
        if line == b"":
            return messages
        length = int(line.decode("ascii").partition(":")[2].strip())
        assert buffer.readline().strip() == b""
        messages.append(json.loads(buffer.read(length)))


def drive(linter: SoftmaxLinter, messages: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    """Run the loop over scripted input; return the exit code and output."""
    out = io.BytesIO()
    code = lsp.serve_forever(linter, stdin=io.BytesIO(b"".join(frame(m) for m in messages)), stdout=out)
    return code, outgoing(out)


def replies(messages: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Server replies keyed by request id (notifications have none)."""
    return {m["id"]: m for m in messages if "id" in m and ("result" in m or "error" in m)}


def notifications(messages: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    """Payloads of one notification type, in order."""
    return [m["params"] for m in messages if m.get("method") == method]


def request(msg_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A JSON-RPC request. ``params`` is omitted when None."""
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def notify(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """A JSON-RPC notification."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


def opened(uri: str, text: str, version: int = 1) -> dict[str, Any]:
    """A didOpen for a Python document."""
    return notify(
        "textDocument/didOpen",
        {"textDocument": {"uri": uri, "languageId": "python", "version": version, "text": text}},
    )


# ------------------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------------------
def test_offsets_map_to_line_and_character() -> None:
    source = "ab\ncde\n"
    starts = lsp.line_starts(source)

    assert lsp.position_at(source, starts, 0) == {"line": 0, "character": 0}
    assert lsp.position_at(source, starts, 4) == {"line": 1, "character": 1}
    assert lsp.position_at(source, starts, len(source)) == {"line": 2, "character": 0}


def test_positions_count_in_utf16_code_units() -> None:
    source = "a\U0001f600b\n"  # the emoji is one code point but two UTF-16 units

    assert lsp.position_at(source, lsp.line_starts(source), 3) == {"line": 0, "character": 4}


# ------------------------------------------------------------------------------
# Diagnostics mapping
# ------------------------------------------------------------------------------
def test_a_finding_maps_to_a_range_severity_and_message(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))

    [diagnostic] = lsp.diagnostics_for(BUGGY_SNIPPET, linter.evaluate_snippet(BUGGY_SNIPPET), linter)

    start = BUGGY_SNIPPET.index(BUGGY_OPERATOR)
    line = BUGGY_SNIPPET.count("\n", 0, start)
    character = start - (BUGGY_SNIPPET.rfind("\n", 0, start) + 1)
    assert diagnostic["range"] == {
        "start": {"line": line, "character": character},
        "end": {"line": line, "character": character + len(BUGGY_OPERATOR)},
    }
    assert diagnostic["severity"] == lsp.SEVERITY_ERROR
    assert diagnostic["source"] == "softmaxlint"
    assert BUGGY_OPERATOR in diagnostic["message"]


def test_a_clean_file_produces_no_diagnostics(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder())

    assert lsp.diagnostics_for(BUGGY_SNIPPET, linter.evaluate_snippet(BUGGY_SNIPPET), linter) == []


# ------------------------------------------------------------------------------
# Server loop
# ------------------------------------------------------------------------------
def test_initialize_advertises_full_sync_and_shuts_down(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder())

    code, out = drive(
        linter,
        [
            request(1, "initialize", {"capabilities": {}}),
            notify("initialized", {}),
            request(2, "shutdown"),
            notify("exit", {}),
        ],
    )

    assert code == 0
    assert replies(out)[1]["result"]["capabilities"] == {"textDocumentSync": 1}
    assert replies(out)[1]["result"]["serverInfo"]["name"] == "softmaxlint"
    assert replies(out)[2]["result"] is None


def test_open_publishes_diagnostics_for_the_file(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))

    code, out = drive(linter, [opened("file:///average.py", BUGGY_SNIPPET), notify("exit", {})])
    (published,) = notifications(out, "textDocument/publishDiagnostics")

    assert code == 0
    assert published["uri"] == "file:///average.py"
    assert published["version"] == 1
    (diagnostic,) = published["diagnostics"]
    assert diagnostic["severity"] == lsp.SEVERITY_ERROR


def test_change_updates_text_but_only_save_relints(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))
    uri = "file:///average.py"
    fixed = BUGGY_SNIPPET.replace(BUGGY_OPERATOR, "<")

    _, out = drive(
        linter,
        [
            opened(uri, fixed),
            notify(
                "textDocument/didChange",
                {"textDocument": {"uri": uri, "version": 2}, "contentChanges": [{"text": BUGGY_SNIPPET}]},
            ),
            notify("textDocument/didSave", {"textDocument": {"uri": uri}}),
            notify("exit", {}),
        ],
    )
    published = notifications(out, "textDocument/publishDiagnostics")

    assert [len(item["diagnostics"]) for item in published] == [0, 1]


def test_close_clears_the_files_diagnostics(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))
    uri = "file:///average.py"

    _, out = drive(
        linter,
        [
            opened(uri, BUGGY_SNIPPET),
            notify("textDocument/didClose", {"textDocument": {"uri": uri}}),
            notify("exit", {}),
        ],
    )
    published = notifications(out, "textDocument/publishDiagnostics")

    assert [len(item["diagnostics"]) for item in published] == [1, 0]


def test_an_unknown_request_gets_method_not_found(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=fim_responder())

    _, out = drive(linter, [request(9, "workspace/unknown"), notify("exit", {})])

    assert replies(out)[9]["error"]["code"] == -32601


def test_a_dead_scoring_server_clears_instead_of_going_stale(
    make_linter: MakeLinter, capsys: pytest.CaptureFixture[str]
) -> None:
    linter = make_linter(error=httpx2.ConnectError("connection refused"))

    _, out = drive(linter, [opened("file:///average.py", BUGGY_SNIPPET), notify("exit", {})])
    (published,) = notifications(out, "textDocument/publishDiagnostics")

    assert published["diagnostics"] == []
    assert "could not reach vLLM" in capsys.readouterr().err


# ------------------------------------------------------------------------------
# `lsp check`: one-shot diagnostics JSON
# ------------------------------------------------------------------------------
def test_lsp_check_prints_diagnostics_json(tmp_path: Path, serve: Serve) -> None:
    serve(fim_responder(surprising=BUGGY_OPERATOR))
    target = tmp_path / "average.py"
    target.write_text(BUGGY_SNIPPET, encoding="utf-8")

    result = runner.invoke(cli.app, ["lsp", "check", str(target)])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    (diagnostic,) = json.loads(result.output)
    assert diagnostic["uri"] == target.resolve().as_uri()
    assert diagnostic["severity"] == lsp.SEVERITY_ERROR
    assert diagnostic["code"] == "CRITICAL"
    assert BUGGY_OPERATOR in diagnostic["message"]


def test_lsp_check_concise_prints_ruff_style_lines(tmp_path: Path, serve: Serve) -> None:
    serve(fim_responder(surprising=BUGGY_OPERATOR))
    target = tmp_path / "average.py"
    target.write_text(BUGGY_SNIPPET, encoding="utf-8")

    result = runner.invoke(cli.app, ["lsp", "check", "--output-format", "concise", str(target)])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    start = BUGGY_SNIPPET.index(BUGGY_OPERATOR)
    line = BUGGY_SNIPPET.count("\n", 0, start) + 1
    character = start - (BUGGY_SNIPPET.rfind("\n", 0, start) + 1) + 1
    assert result.output.strip().startswith(f"{target}:{line}:{character}: [CRITICAL]")
    assert BUGGY_OPERATOR in result.output


def test_lsp_check_concise_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: StubServer
) -> None:
    """`OUTPUT_FORMAT` is read like any other setting: after the env is set."""
    monkeypatch.setenv("OUTPUT_FORMAT", "concise")
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))
    server.responder = fim_responder(surprising=BUGGY_OPERATOR)

    class _StubbedLinter(SoftmaxLinter):
        def __init__(self, config: Settings | None = None) -> None:
            super().__init__(config)
            self.client = server.client()

    monkeypatch.setattr(cli, "SoftmaxLinter", _StubbedLinter)
    target = tmp_path / "average.py"
    target.write_text(BUGGY_SNIPPET, encoding="utf-8")

    result = runner.invoke(cli.app, ["lsp", "check", str(target)])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    assert result.output.strip().startswith(f"{target}:")


def test_lsp_check_rejects_an_unknown_output_format(tmp_path: Path, serve: Serve) -> None:
    serve(fim_responder())
    target = tmp_path / "average.py"
    target.write_text(BUGGY_SNIPPET, encoding="utf-8")

    result = runner.invoke(cli.app, ["lsp", "check", "--output-format", "sarif", str(target)])

    assert result.exit_code == 2


def test_lsp_check_is_quiet_and_exits_zero_on_clean_code(tmp_path: Path, serve: Serve) -> None:
    serve(fim_responder())
    target = tmp_path / "average.py"
    target.write_text(BUGGY_SNIPPET, encoding="utf-8")

    result = runner.invoke(cli.app, ["lsp", "check", str(target)])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert json.loads(result.output) == []


def test_lsp_check_ranges_point_at_the_written_token(serve: Serve) -> None:
    """The corpus operator's range covers the exact slice the server scored."""
    serve(fim_responder(surprising="==", alternative="zzz"))
    target = BUG_DIR / "operators.py"

    result = runner.invoke(cli.app, ["lsp", "check", str(target)])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    lines = target.read_text(encoding="utf-8").splitlines()
    hits = [item for item in json.loads(result.output) if item["code"] == "CRITICAL" and "==" in item["message"]]
    assert hits, "expected the seeded == to be flagged"
    for diagnostic in hits:
        start = diagnostic["range"]["start"]
        end = diagnostic["range"]["end"]
        assert start["line"] == end["line"]
        rest = lines[start["line"]][start["character"] :]
        assert rest.startswith(("==", " ==")), rest
