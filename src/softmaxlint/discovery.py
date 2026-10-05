"""Turn the paths a user typed into the list of Python files worth linting.

Ignore rules follow git: the ``.gitignore`` at the enclosing work tree root
plus every nested ``.gitignore`` found while descending, each applied relative
to the directory that declared it. A handful of directories cannot be expressed
in a ``.gitignore`` at all (``.git/`` itself, tool caches, virtualenvs) and are
skipped unconditionally.
"""

from typing import TYPE_CHECKING

import pathspec

from .errors import DiscoveryError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

#: Always skipped, regardless of ``.gitignore``. Anything a user would want to
#: lint inside these is a mistake rather than an intent.
ALWAYS_SKIPPED_DIRS = frozenset(
    {
        ".eggs",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)

#: (directory the patterns are relative to, its compiled patterns)
type IgnoreStack = tuple[tuple[Path, pathspec.PathSpec], ...]


def find_work_tree_root(start: Path) -> Path | None:
    """Return the nearest ancestor of ``start`` containing ``.git``, else ``None``."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _compile(gitignore: Path) -> pathspec.PathSpec | None:
    """Compile one ``.gitignore``. Returns ``None`` if it cannot be read."""
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except OSError, UnicodeDecodeError:
        return None
    return pathspec.PathSpec.from_lines("gitignore", lines)


def _load_ignore_stack(directory: Path, stack: IgnoreStack) -> IgnoreStack:
    """Extend ``stack`` with ``directory/.gitignore`` if there is one."""
    spec = _compile(directory / ".gitignore")
    if spec is None:
        return stack
    return (*stack, (directory, spec))


def _is_ignored(candidate: Path, stack: IgnoreStack, *, is_dir: bool) -> bool:
    """Report whether any active ignore rule excludes ``candidate``.

    A directory is tested with a trailing slash as well, because gitignore's
    directory-only patterns (``build/``) and its recursive ones (``logs/**``)
    only match the ``name/`` spelling.
    """
    for base, spec in stack:
        try:
            relative = candidate.relative_to(base).as_posix()
        except ValueError:
            # The rule was declared outside this subtree; it cannot apply.
            continue
        if is_dir and spec.match_file(f"{relative}/"):
            return True
        if spec.match_file(relative):
            return True
    return False


def _root_stack(directory: Path, *, respect_gitignore: bool) -> IgnoreStack:
    """Seed the ignore stack for a directory the user asked us to walk."""
    if not respect_gitignore:
        return ()
    work_tree = find_work_tree_root(directory)
    # Outside a work tree the only sensible ruleset is the local .gitignore.
    return _load_ignore_stack(work_tree or directory, ())


def _walk(directory: Path, stack: IgnoreStack, *, respect_gitignore: bool) -> Iterator[Path]:
    """Yield lintable files under ``directory`` in a deterministic order."""
    pending: list[tuple[Path, IgnoreStack]] = [(directory, stack)]
    while pending:
        current, inherited = pending.pop()
        # Descending into a directory picks up the .gitignore it may declare.
        active = _load_ignore_stack(current, inherited) if respect_gitignore else inherited

        for entry in sorted(current.iterdir()):
            # Symlinked directories are not descended into: they can loop.
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in ALWAYS_SKIPPED_DIRS or _is_ignored(entry, active, is_dir=True):
                    continue
                pending.append((entry, active))
            elif entry.suffix == ".py" and entry.is_file() and not _is_ignored(entry, active, is_dir=False):
                yield entry


def collect_python_files(paths: Sequence[Path], *, respect_gitignore: bool = True) -> list[Path]:
    """Resolve ``paths`` into a sorted, de-duplicated list of Python files.

    An explicitly named file is always linted, even when ``.gitignore`` would
    exclude it -- the user asked for that exact file. Ignore rules still apply
    to everything discovered by walking an explicitly named directory.

    Raises:
        DiscoveryError: if a path does not exist or is neither a regular
            file nor a directory. An unreadable directory propagates its
            OSError instead; only requested paths get DiscoveryError.
    """
    found: dict[Path, None] = {}

    for path in paths:
        if not path.exists():
            msg = f"path does not exist: {path}"
            raise DiscoveryError(msg)

        if path.is_file():
            found[path] = None
            continue

        if not path.is_dir():
            msg = f"not a regular file or directory: {path}"
            raise DiscoveryError(msg)

        for candidate in _walk(
            path,
            _root_stack(path, respect_gitignore=respect_gitignore),
            respect_gitignore=respect_gitignore,
        ):
            found[candidate] = None

    return sorted(found)
