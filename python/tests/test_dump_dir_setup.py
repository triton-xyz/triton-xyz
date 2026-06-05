import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_backend_module(module: str, dump_dir: Path) -> None:
    env = os.environ.copy()
    env["TT_XYZ_ENABLE_DUMP_DIR"] = str(dump_dir)
    subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_compiler_import_creates_dump_root(tmp_path):
    dump_dir = tmp_path / "compiler_dump_root"
    assert not dump_dir.exists()
    _run_backend_module("backend.compiler", dump_dir)
    assert dump_dir.is_dir()


def test_driver_import_creates_dump_root(tmp_path):
    dump_dir = tmp_path / "driver_dump_root"
    assert not dump_dir.exists()
    _run_backend_module("backend.driver", dump_dir)
    assert dump_dir.is_dir()
