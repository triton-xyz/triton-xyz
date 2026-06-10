#!/usr/bin/env python3
"""Regenerate FileCheck lines for local MLIR lit tests.

The script reads RUN lines from .mlir tests, executes the pipeline before the
final FileCheck command, then feeds that output to tools/agent/generate-test-checks.py.
Negative RUN lines that start with `not` are intentionally skipped.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TEST_PATHS = ("test",)
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = Path("tools/agent/generate-test-checks.py")
MLIR_SOURCE_DELIM_RE = (
    r"^(?!\s*//)(?!\s*(tt\.func|func\.func|llvm\.func)\s+private\b)\s*"
    r"(tt\.func|func\.func|llvm\.func)\b"
)
LLVM_SOURCE_DELIM_RE = r"^\s*module\b"
RUN_RE = re.compile(r"^\s*//\s*RUN:\s?(.*)$")
COMMENT_RE = re.compile(r"^\s*//\s?(.*)$")
LOCAL_EXE_DIRS = (Path("build/bin"),)


@dataclass(frozen=True)
class RunLine:
    command: str
    line_no: int


@dataclass(frozen=True)
class FileCheckRun:
    file: Path
    line_no: int
    command: str
    producer_tokens: list[str]
    prefixes: list[str]
    source_delim_regex: str


def strip_continuation(text: str) -> tuple[str, bool]:
    text = text.rstrip()
    if text.endswith("\\"):
        return text[:-1].rstrip(), True
    return text, False


def collect_run_lines(path: Path) -> list[RunLine]:
    runs: list[RunLine] = []
    pending: list[str] = []
    pending_line_no = 0
    continuing = False

    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        run_match = RUN_RE.match(line)
        if run_match:
            if pending and not continuing:
                runs.append(RunLine(" ".join(pending).strip(), pending_line_no))
                pending = []

            chunk, continuing = strip_continuation(run_match.group(1))
            if not pending:
                pending_line_no = line_no
            pending.append(chunk)
            if not continuing:
                runs.append(RunLine(" ".join(pending).strip(), pending_line_no))
                pending = []
            continue

        if continuing:
            comment_match = COMMENT_RE.match(line)
            if comment_match is None:
                raise ValueError(f"{path}:{pending_line_no}: RUN continuation reaches a non-comment line at {line_no}")
            chunk, continuing = strip_continuation(comment_match.group(1))
            pending.append(chunk)
            if not continuing:
                runs.append(RunLine(" ".join(pending).strip(), pending_line_no))
                pending = []

    if pending:
        if continuing:
            raise ValueError(f"{path}:{pending_line_no}: unterminated RUN continuation")
        runs.append(RunLine(" ".join(pending).strip(), pending_line_no))

    return runs


def split_shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|")
    lexer.whitespace_split = True
    return list(lexer)


def is_filecheck_token(token: str) -> bool:
    return os.path.basename(token) == "FileCheck"


def split_filecheck_run(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    for index in range(len(tokens) - 2, -1, -1):
        if tokens[index] == "|" and is_filecheck_token(tokens[index + 1]):
            return tokens[:index], tokens[index + 1 :]
    return None


def extract_check_prefixes(filecheck_tokens: list[str]) -> list[str]:
    prefixes: list[str] = []
    index = 1
    while index < len(filecheck_tokens):
        token = filecheck_tokens[index]
        value = None

        if token.startswith("--check-prefix="):
            value = token.split("=", 1)[1]
        elif token == "--check-prefix" and index + 1 < len(filecheck_tokens):
            index += 1
            value = filecheck_tokens[index]
        elif token.startswith("--check-prefixes="):
            value = token.split("=", 1)[1]
        elif token == "--check-prefixes" and index + 1 < len(filecheck_tokens):
            index += 1
            value = filecheck_tokens[index]

        if value:
            for prefix in value.split(","):
                if prefix and prefix not in prefixes:
                    prefixes.append(prefix)

        index += 1

    return prefixes or ["CHECK"]


def has_llvm_ir_translation(tokens: Iterable[str]) -> bool:
    return any(token == "-mlir-to-llvmir" for token in tokens)


def substitute_lit_token(token: str, test_file: Path) -> str:
    source = str(test_file)
    source_dir = str(test_file.parent)
    return token.replace("%s", source).replace("%S", source_dir).replace("%p", source_dir).replace("%%", "%")


def resolve_executable(token: str) -> str:
    if token == "|" or "/" in token or shutil.which(token):
        return token

    for directory in LOCAL_EXE_DIRS:
        candidate = directory / token
        candidate_path = REPO_ROOT / candidate
        if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
            return str(candidate)

    return token


def resolve_pipeline_executables(tokens: list[str]) -> list[str]:
    resolved: list[str] = []
    next_is_command = True
    for token in tokens:
        if token == "|":
            resolved.append(token)
            next_is_command = True
            continue

        resolved.append(resolve_executable(token) if next_is_command else token)
        next_is_command = False

    return resolved


def shell_join(tokens: Iterable[str]) -> str:
    parts: list[str] = []
    for token in tokens:
        if token == "|":
            parts.append("|")
        else:
            parts.append(shlex.quote(token))
    return " ".join(parts)


def parse_filecheck_run(path: Path, run: RunLine) -> FileCheckRun | None:
    tokens = split_shell_tokens(run.command)
    if not tokens:
        return None

    if tokens[0] == "not":
        return None

    split_run = split_filecheck_run(tokens)
    if split_run is None:
        return None

    producer_tokens, filecheck_tokens = split_run
    if not producer_tokens:
        raise ValueError(f"{path}:{run.line_no}: empty producer before FileCheck")

    source_delim_regex = LLVM_SOURCE_DELIM_RE if has_llvm_ir_translation(producer_tokens) else MLIR_SOURCE_DELIM_RE

    return FileCheckRun(
        file=path,
        line_no=run.line_no,
        command=run.command,
        producer_tokens=resolve_pipeline_executables([substitute_lit_token(token, path) for token in producer_tokens]),
        prefixes=extract_check_prefixes(filecheck_tokens),
        source_delim_regex=source_delim_regex,
    )


def unique_sorted(paths: Iterable[Path]) -> list[Path]:
    normalized = [Path(os.path.normpath(str(path))) for path in paths]
    return sorted(dict.fromkeys(normalized))


def collect_tests_from_path(path: Path, *, ignore_non_mlir: bool = False) -> list[Path]:
    tests: list[Path] = []
    if path.is_dir():
        tests.extend(path.rglob("*.mlir"))
    elif path.is_file():
        if path.suffix != ".mlir":
            if ignore_non_mlir:
                return []
            raise ValueError(f"{path}: expected an .mlir test file")
        tests.append(path)
    else:
        raise FileNotFoundError(path)
    return unique_sorted(tests)


def normalize_selector(selector: Path) -> str:
    text = str(selector).replace("\\", "/").rstrip("/")
    while text.startswith("./"):
        text = text[2:]
    return text


def path_candidates(path: Path, roots: Iterable[Path]) -> set[str]:
    candidates = {
        path.as_posix(),
        path.resolve().as_posix(),
        path.name,
        path.stem,
    }
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        candidates.update({relative.as_posix(), relative.name, relative.stem})
    return candidates


def parent_candidates(path: Path, roots: Iterable[Path]) -> set[str]:
    candidates: set[str] = set()
    for parent in path.parents:
        if parent == Path("."):
            continue
        candidates.update({parent.as_posix(), parent.resolve().as_posix(), parent.name})
        for root in roots:
            try:
                relative = parent.relative_to(root)
            except ValueError:
                continue
            if relative != Path("."):
                candidates.add(relative.as_posix())
    return candidates


def candidate_matches(selector: str, candidate: str) -> bool:
    selector = selector.casefold()
    candidate = candidate.casefold()
    if glob.has_magic(selector):
        return fnmatch.fnmatchcase(candidate, selector)
    return candidate == selector or candidate.endswith("/" + selector)


def selector_matches_test(selector: Path, test: Path, roots: Iterable[Path]) -> bool:
    text = normalize_selector(selector)
    candidates = path_candidates(test, roots) | parent_candidates(test, roots)
    return any(candidate_matches(text, candidate) for candidate in candidates)


def discover_default_tests(roots: Iterable[Path]) -> list[Path]:
    tests: list[Path] = []
    for root in roots:
        tests.extend(collect_tests_from_path(root))
    return unique_sorted(tests)


def discover_tests(selectors: list[Path]) -> list[Path]:
    roots = [Path(path) for path in DEFAULT_TEST_PATHS]
    default_tests = discover_default_tests(roots)
    if not selectors:
        return default_tests

    tests: list[Path] = []
    for selector in selectors:
        matched: list[Path] = []
        if selector.exists():
            matched.extend(collect_tests_from_path(selector))
        elif glob.has_magic(str(selector)):
            for path in glob.glob(str(selector), recursive=True):
                matched.extend(collect_tests_from_path(Path(path), ignore_non_mlir=True))

        matched.extend(test for test in default_tests if selector_matches_test(selector, test, roots))
        if not matched:
            raise FileNotFoundError(f"{selector}: no .mlir tests matched")
        tests.extend(matched)

    return unique_sorted(tests)


def run_command(command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REPO_ROOT,
    )


def run_producer(run: FileCheckRun) -> str:
    command = shell_join(run.producer_tokens)
    completed = run_command(["bash", "-o", "pipefail", "-c", command])
    if completed.returncode != 0:
        raise RuntimeError(
            f"{run.file}:{run.line_no}: producer failed with exit code {completed.returncode}\n{completed.stderr}"
        )
    return completed.stdout


def generator_command(run: FileCheckRun, prefix: str, inplace: bool) -> list[str]:
    command = [
        str(GENERATOR),
        "--source_delim_regex",
        run.source_delim_regex,
        # "--strict_name_re",
        # "1",
        "--check-prefix",
        prefix,
        "--source",
        str(run.file),
    ]
    if inplace:
        command.insert(1, "-i")
    return command


def run_generator(run: FileCheckRun, prefix: str, input_text: str, inplace: bool) -> None:
    completed = run_command(generator_command(run, prefix, inplace), input_text=input_text)
    if completed.returncode != 0:
        stderr = completed.stderr or completed.stdout
        raise RuntimeError(
            f"{run.file}:{run.line_no}: generate-test-checks.py failed for "
            f"{prefix} with exit code {completed.returncode}\n{stderr}"
        )


def update_run(run: FileCheckRun, dry_run: bool) -> None:
    producer_output = run_producer(run)
    for prefix in run.prefixes:
        run_generator(run, prefix, producer_output, inplace=False)
        if not dry_run:
            run_generator(run, prefix, producer_output, inplace=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Optional MLIR test files, directories, globs, or path/name fragments to process. "
            "With no arguments, defaults to `test`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run producers and validate check generation without editing files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors.",
    )
    args = parser.parse_args()

    if not (REPO_ROOT / GENERATOR).exists():
        print(f"error: missing {GENERATOR}", file=sys.stderr)
        return 1

    tests = discover_tests(args.paths)
    processed = 0
    skipped = 0
    failures = 0

    for test in tests:
        try:
            runs = collect_run_lines(test)
            for run in runs:
                parsed = parse_filecheck_run(test, run)
                if parsed is None:
                    skipped += 1
                    if not args.quiet:
                        print(f"skip {test}:{run.line_no}: {run.command}")
                    continue

                update_run(parsed, dry_run=args.dry_run)
                processed += 1
                if not args.quiet:
                    prefixes = ",".join(parsed.prefixes)
                    mode = "validated" if args.dry_run else "updated"
                    print(f"{mode} {test}:{run.line_no} [{prefixes}]")
        except Exception as error:
            failures += 1
            print(f"error: {error}", file=sys.stderr)

    if not args.quiet:
        mode = "validated" if args.dry_run else "updated"
        print(f"{mode}: {processed}, skipped: {skipped}, failed: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
