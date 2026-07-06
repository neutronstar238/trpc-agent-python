# Tencent is pleased to support the open source community by making tRPC-Agent-Python available.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# tRPC-Agent-Python is licensed under Apache-2.0.
"""Repository-wide pytest configuration for optional dependency suites."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent

OPTIONAL_TEST_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code_executors/cube", ("e2b_code_interpreter",)),
    ("memory/test_mempalace_memory_service.py", ("mempalace",)),
    ("tools/test_mempalace_tool.py", ("mempalace",)),
    ("server/a2a", ("a2a",)),
    ("server/ag_ui", ("ag_ui",)),
    ("server/agents/claude", ("claude_agent_sdk",)),
    ("server/openclaw", ("nanobot", "telegram", "aiofiles")),
)


def _missing_modules(modules: Iterable[str]) -> list[str]:
    return [name for name in modules if importlib.util.find_spec(name) is None]


def _relative_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def pytest_ignore_collect(collection_path, config):  # noqa: ANN001 - pytest hook signature.
    rel = _relative_posix(Path(collection_path))
    for prefix, modules in OPTIONAL_TEST_PREFIXES:
        if rel == prefix or rel.startswith(prefix.rstrip("/") + "/"):
            if _missing_modules(modules):
                return True
    return False


def pytest_report_header(config):  # noqa: ANN001 - pytest hook signature.
    ignored = []
    for prefix, modules in OPTIONAL_TEST_PREFIXES:
        missing = _missing_modules(modules)
        if missing:
            ignored.append(f"{prefix} (missing: {', '.join(missing)})")
    if not ignored:
        return []
    return ["optional dependency test suites ignored: " + "; ".join(ignored)]
