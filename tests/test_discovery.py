"""Turning requested paths into the files that actually get linted."""

import os
from pathlib import Path

import pytest

from softmaxlint.discovery import collect_python_files, find_work_tree_root
from softmaxlint.errors import DiscoveryError


def write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway work tree: `.git/` plus a handful of Python files."""
    (tmp_path / ".git").mkdir()
    write(tmp_path / "a.py")
    write(tmp_path / "b.py")
    write(tmp_path / "notes.md")
    return tmp_path


def relative(files: list[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in files]


# ------------------------------------------------------------------------------
# Traversal
# ------------------------------------------------------------------------------
def test_directories_are_walked_recursively(project: Path) -> None:
    write(project / "pkg" / "__init__.py")
    write(project / "pkg" / "sub" / "deep.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py", "pkg/__init__.py", "pkg/sub/deep.py"]


def test_only_python_files_are_collected(project: Path) -> None:
    write(project / "script.sh")
    write(project / "types.pyi")
    write(project / "pkg" / "data.json")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_results_are_sorted_so_output_is_stable(project: Path) -> None:
    write(project / "z.py")
    write(project / "m" / "c.py")

    files = collect_python_files([project])

    assert files == sorted(files)


def test_a_file_named_directly_is_collected_without_walking(project: Path) -> None:
    files = collect_python_files([project / "a.py"])

    assert files == [project / "a.py"]


def test_a_dir_and_a_file_inside_it_are_deduplicated(project: Path) -> None:
    files = collect_python_files([project, project / "a.py"])

    assert files.count(project / "a.py") == 1


def test_tool_and_vcs_directories_are_skipped(project: Path) -> None:
    """These cannot be excluded from a .gitignore, so they are skipped outright."""
    for skipped in (".git", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache", "node_modules"):
        write(project / skipped / "hidden.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_build_and_dist_are_skipped_only_when_gitignored(project: Path) -> None:
    """`build/` is ordinary as far as git is concerned, so .gitignore decides."""
    write(project / "build" / "generated.py")
    assert "build/generated.py" in relative(collect_python_files([project]), project)

    write(project / ".gitignore", "build/\n")
    assert "build/generated.py" not in relative(collect_python_files([project]), project)

    assert "build/generated.py" in relative(collect_python_files([project], respect_gitignore=False), project)


# ------------------------------------------------------------------------------
# .gitignore handling
# ------------------------------------------------------------------------------
def test_gitignored_files_are_excluded(project: Path) -> None:
    write(project / "scratch.py")
    write(project / ".gitignore", "scratch.py\n")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_a_leading_slash_anchors_the_pattern_to_the_ignores_own_directory(project: Path) -> None:
    write(project / ".gitignore", "/a.py\n")
    write(project / "pkg" / "a.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["b.py", "pkg/a.py"]


def test_a_bare_name_matches_at_any_depth(project: Path) -> None:
    write(project / ".gitignore", "a.py\n")
    write(project / "pkg" / "a.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["b.py"]


def test_a_negation_re_includes_a_file(project: Path) -> None:
    write(project / ".gitignore", "*.py\n!a.py\n!b.py\n")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_a_double_star_pattern_matches_across_directories(project: Path) -> None:
    write(project / ".gitignore", "logs/**/*.py\n")
    write(project / "logs" / "deep" / "run.py")
    write(project / "logs" / "top.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_a_directory_only_pattern_prunes_the_whole_subtree(project: Path) -> None:
    write(project / ".gitignore", "vendor/\n")
    write(project / "vendor" / "deep" / "nested.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py"]


def test_a_nested_gitignore_applies_relative_to_its_own_directory(project: Path) -> None:
    write(project / "pkg" / ".gitignore", "private.py\n")
    write(project / "pkg" / "private.py")
    write(project / "pkg" / "public.py")
    write(project / "private.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py", "pkg/public.py", "private.py"]


def test_root_and_nested_gitignores_combine(project: Path) -> None:
    write(project / ".gitignore", "*.log.py\n")
    write(project / "pkg" / ".gitignore", "draft.py\n")
    write(project / "pkg" / "draft.py")
    write(project / "pkg" / "public.py")
    write(project / "pkg" / "changelog.log.py")
    write(project / "notes.log.py")

    files = collect_python_files([project])

    assert relative(files, project) == ["a.py", "b.py", "pkg/public.py"]


def test_an_unreadable_nested_gitignore_does_not_abort_the_walk(project: Path) -> None:
    """A .gitignore we cannot read must not cost us the entire run."""
    write(project / "pkg" / "public.py")
    (project / "pkg" / ".gitignore").write_bytes(b"\xff\xfe\x00invalid")

    files = collect_python_files([project])

    assert "pkg/public.py" in relative(files, project)


def test_an_explicitly_named_ignored_file_is_still_collected(project: Path) -> None:
    write(project / "scratch.py")
    write(project / ".gitignore", "scratch.py\n")

    files = collect_python_files([project / "scratch.py"])

    assert files == [project / "scratch.py"]


def test_ignores_from_the_enclosing_work_tree_apply_to_a_subdirectory(project: Path) -> None:
    """Walking `pkg/` must still honour the root .gitignore above it."""
    write(project / ".gitignore", "generated.py\n")
    write(project / "pkg" / "generated.py")
    write(project / "pkg" / "handwritten.py")

    files = collect_python_files([project / "pkg"])

    assert relative(files, project) == ["pkg/handwritten.py"]


def test_a_gitignore_outside_any_work_tree_is_used_directly(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    write(tmp_path / "ignored.py")
    write(tmp_path / "kept.py")

    files = collect_python_files([tmp_path])

    assert relative(files, tmp_path) == ["kept.py"]


def test_disabling_gitignore_collects_everything(project: Path) -> None:
    write(project / ".gitignore", "a.py\nvendor/\n")
    write(project / "vendor" / "v.py")

    files = collect_python_files([project], respect_gitignore=False)

    assert relative(files, project) == ["a.py", "b.py", "vendor/v.py"]


# ------------------------------------------------------------------------------
# Work tree detection
# ------------------------------------------------------------------------------
def test_the_work_tree_root_is_found_by_walking_up(project: Path) -> None:
    write(project / "pkg" / "deep" / "leaf.py")

    assert find_work_tree_root(project / "pkg" / "deep") == project


def test_no_work_tree_root_is_reported_outside_a_repository(tmp_path: Path) -> None:
    assert find_work_tree_root(tmp_path) is None


# ------------------------------------------------------------------------------
# Error branches
# ------------------------------------------------------------------------------
def test_a_missing_path_is_an_error(project: Path) -> None:
    with pytest.raises(DiscoveryError, match="does not exist"):
        collect_python_files([project / "nope.py"])


def test_a_path_that_is_neither_file_nor_directory_is_an_error(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(DiscoveryError, match="not a regular file or directory"):
        collect_python_files([fifo])


def test_a_dangling_symlink_is_reported_as_missing(tmp_path: Path) -> None:
    dangling = tmp_path / "gone.py"
    dangling.symlink_to(tmp_path / "nowhere.py")

    with pytest.raises(DiscoveryError, match="does not exist"):
        collect_python_files([dangling])
