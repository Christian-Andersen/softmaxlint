"""Language Server Protocol support: one-shot diagnostics and a stdio server.

The wire format is plain LSP over stdio (``Content-Length`` framing, JSON-RPC
2.0), so any editor with an LSP client works -- Neovim, VS Code, Helix -- with
no per-editor code here. Only this subset of the protocol is implemented:

* ``initialize`` / ``initialized`` / ``shutdown`` / ``exit``
* ``textDocument/didOpen`` and ``textDocument/didSave`` trigger a lint;
  ``textDocument/didChange`` only updates the stored text
* ``textDocument/didClose`` drops the document and clears its diagnostics
* ``textDocument/publishDiagnostics`` carries the findings

Full document sync is advertised (``TextDocumentSyncKind.Full``). Linting runs
only on open and save: a full FIM pass over a file takes tens of seconds
against the local server, so linting on every keystroke would queue work
faster than the server drains it. Positions use the default UTF-16 encoding.

Lint failures (an unreachable scoring server) clear the file's diagnostics
and are logged to stderr, where the editor's LSP log picks them up.
"""

from __future__ import annotations

import bisect
import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import SoftmaxLintError

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

    from .linter import AnomalyLevel, SoftmaxLinter, TokenAnalysis

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""Anything JSON can carry; keeps wire handling typed without bare ``Any``."""

SEVERITY_ERROR = 1
SEVERITY_WARNING = 2

_SEVERITIES: dict[AnomalyLevel, int] = {"CRITICAL": SEVERITY_ERROR, "WARNING": SEVERITY_WARNING}


def line_starts(source: str) -> list[int]:
    """Character offset where each line begins, starting with 0."""
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def utf16_units(text: str) -> int:
    """Length of ``text`` in UTF-16 code units, the LSP position unit."""
    return len(text.encode("utf-16-le")) // 2


def position_at(source: str, starts: list[int], offset: int) -> dict[str, int]:
    """Map a character ``offset`` to an LSP position."""
    line = bisect.bisect_right(starts, offset) - 1
    return {"line": line, "character": utf16_units(source[starts[line] : offset])}


def diagnostic_for(source: str, starts: list[int], level: AnomalyLevel, analysis: TokenAnalysis) -> dict[str, Any]:
    """One LSP diagnostic for a token graded ``level``."""
    actual_pct = analysis["actual_prob"] * 100
    top_pct = analysis["top_prob"] * 100
    return {
        "range": {
            "start": position_at(source, starts, analysis["start"]),
            "end": position_at(source, starts, analysis["end"]),
        },
        "severity": _SEVERITIES[level],
        "source": "softmaxlint",
        "code": level,
        "message": (
            f"Wrote {analysis['token']!r} ({actual_pct:.2f}%) "
            f"but the model expected {analysis['top_token']!r} ({top_pct:.2f}%)"
        ),
    }


def diagnostics_for(source: str, results: list[TokenAnalysis], linter: SoftmaxLinter) -> list[dict[str, Any]]:
    """Every finding in ``source``, as LSP diagnostics without uris."""
    starts = line_starts(source)
    graded = [(linter.classify(res), res) for res in results]
    return [diagnostic_for(source, starts, level, res) for level, res in graded if level is not None]


def file_uri(path: Path) -> str:
    """Absolute file uri for ``path``."""
    return path.resolve().as_uri()


def format_concise(path: str, diagnostic: dict[str, Any]) -> str:
    """One ``path:line:col: [CODE] message`` line, ruff-concise style.

    Lines and columns are 1-based for display; the diagnostic range stays
    0-based LSP.
    """
    start = diagnostic["range"]["start"]
    code = diagnostic.get("code")
    shown = f"[{code}] " if code else ""
    return f"{path}:{start['line'] + 1}:{start['character'] + 1}: {shown}{diagnostic['message']}"


class _FramingError(Exception):
    """A message that is not JSON-RPC; the loop skips it."""


def _read_exactly(stream: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(stream: BinaryIO) -> JsonValue:
    """Read one framed message. Raises ``EOFError`` on a closed stream."""
    first = stream.readline()
    if first == b"":
        raise EOFError
    length = 0
    line = first
    while True:
        text = line.decode("ascii", errors="replace").strip()
        if not text:
            break
        name, _, value = text.partition(":")
        if name.strip().lower() == "content-length":
            try:
                length = int(value.strip())
            except ValueError as exc:
                msg = f"bad Content-Length: {value!r}"
                raise _FramingError(msg) from exc
        line = stream.readline()
        if line == b"":
            raise EOFError
    try:
        return json.loads(_read_exactly(stream, length))
    except json.JSONDecodeError as exc:
        msg = f"not JSON: {exc}"
        raise _FramingError(msg) from exc


def _send(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def _respond(stream: BinaryIO, msg_id: JsonValue, result: JsonValue) -> None:
    _send(stream, {"jsonrpc": "2.0", "id": msg_id, "result": result})


def _fail(stream: BinaryIO, msg_id: JsonValue, code: int, message: str) -> None:
    _send(stream, {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def _notify(stream: BinaryIO, method: str, params: dict[str, Any]) -> None:
    _send(stream, {"jsonrpc": "2.0", "method": method, "params": params})


def _lint_and_publish(
    linter: SoftmaxLinter,
    stdout: BinaryIO,
    uri: str,
    text: str,
    version: int | None,
) -> None:
    """Score ``text`` and publish its diagnostics, clearing them on failure."""
    try:
        results = linter.evaluate_snippet(text)
    except SoftmaxLintError as exc:
        print(f"softmaxlint: cannot score {uri}: {exc}", file=sys.stderr)
        results = []
    params: dict[str, Any] = {"uri": uri, "diagnostics": diagnostics_for(text, results, linter)}
    if version is not None:
        params["version"] = version
    _notify(stdout, "textDocument/publishDiagnostics", params)


@dataclass
class _Session:
    """Documents open in the editor, and the linter scoring them."""

    linter: SoftmaxLinter
    stdout: BinaryIO
    texts: dict[str, str] = field(default_factory=dict)
    versions: dict[str, int | None] = field(default_factory=dict)

    def notification(self, method: JsonValue, params: JsonValue) -> None:
        """Handle one client notification; unknown ones are ignored."""
        if not isinstance(params, dict):
            return
        if method == "textDocument/didOpen":
            self.opened(_mapping(params.get("textDocument")))
        elif method == "textDocument/didChange":
            doc = _mapping(params.get("textDocument"))
            self.changed(doc, params.get("contentChanges"))
        elif method == "textDocument/didSave":
            self.saved(_mapping(params.get("textDocument")), params.get("text"))
        elif method == "textDocument/didClose":
            self.closed(_mapping(params.get("textDocument")).get("uri"))

    def opened(self, doc: dict[str, JsonValue]) -> None:
        """Store a document and lint it."""
        uri, text = doc.get("uri"), doc.get("text")
        if not isinstance(uri, str) or not isinstance(text, str):
            return
        self.texts[uri] = text
        version = doc.get("version")
        self.versions[uri] = version if isinstance(version, int) else None
        _lint_and_publish(self.linter, self.stdout, uri, text, self.versions[uri])

    def changed(self, doc: dict[str, JsonValue], changes: JsonValue) -> None:
        """Store the latest text. Linting waits for open or save."""
        uri = doc.get("uri")
        if not isinstance(uri, str) or uri not in self.texts or not isinstance(changes, list) or not changes:
            return
        text = _mapping(changes[-1]).get("text")
        if not isinstance(text, str):
            return
        self.texts[uri] = text
        version = doc.get("version")
        if isinstance(version, int):
            self.versions[uri] = version

    def saved(self, doc: dict[str, JsonValue], text: JsonValue) -> None:
        """Lint the stored text, preferring one the client sends along."""
        uri = doc.get("uri")
        if not isinstance(uri, str) or uri not in self.texts:
            return
        if isinstance(text, str):
            self.texts[uri] = text
        _lint_and_publish(self.linter, self.stdout, uri, self.texts[uri], self.versions[uri])

    def closed(self, uri: JsonValue) -> None:
        """Forget a document and clear its diagnostics."""
        if not isinstance(uri, str):
            return
        self.texts.pop(uri, None)
        self.versions.pop(uri, None)
        _notify(self.stdout, "textDocument/publishDiagnostics", {"uri": uri, "diagnostics": []})


def serve_forever(
    linter: SoftmaxLinter,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> int:
    """Speak LSP on stdio until ``exit``. Returns the process exit code."""
    instream = stdin if stdin is not None else sys.stdin.buffer
    outstream = stdout if stdout is not None else sys.stdout.buffer
    session = _Session(linter, outstream)

    while True:
        try:
            message = _read_message(instream)
        except EOFError:
            return 0
        except _FramingError as exc:
            print(f"softmaxlint: skipping malformed message: {exc}", file=sys.stderr)
            continue
        if not isinstance(message, dict):
            continue
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        if method == "exit":
            return 0
        if msg_id is None:
            session.notification(method, params)
            continue
        if method == "initialize":
            _respond(
                outstream,
                msg_id,
                {
                    "capabilities": {"textDocumentSync": 1},
                    "serverInfo": {"name": "softmaxlint"},
                },
            )
        elif method == "shutdown":
            _respond(outstream, msg_id, None)
        else:
            _fail(outstream, msg_id, -32601, f"method not found: {method}")


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    """``value`` as an object, or empty when it is anything else."""
    return value if isinstance(value, dict) else {}
