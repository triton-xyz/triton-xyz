import lit.TestingConfig
import lit.formats
import os
import shutil
import subprocess
import tempfile

# for type hint
config = config  # noqa: F821
config: lit.TestingConfig.TestingConfig

config.name = "triton-xyz"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = set([".mlir"])

config.test_exec_root = os.path.join(
    tempfile.gettempdir(),
    f"lit-{os.getenv('USER', '')}",
)
# config.test_exec_root = os.path.join("_demos", "lit") # debug

# `FileCheck` or `filecheck` from https://github.com/AntonLydike/filecheck
filecheck_executable = "FileCheck"
if not shutil.which(filecheck_executable):
    filecheck_executable = "filecheck"
    config.substitutions.append(("FileCheck", filecheck_executable))


def _tool_supports_arg(tool: str, arg: str) -> bool:
    tool_path = shutil.which(tool)
    if not tool_path:
        return False
    try:
        result = subprocess.run(
            [tool_path, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return arg in result.stdout or arg in result.stderr


if _tool_supports_arg("triton-xyz-opt", "--proton-to-xyz"):
    config.available_features.add("triton_xyz_build_proton")
