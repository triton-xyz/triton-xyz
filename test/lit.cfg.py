import lit.TestingConfig
import lit.formats
import os
import shutil
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
