import functools
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, llvm, passes  # ty:ignore[unresolved-import]
from triton.runtime.build import _build

_DUMP_INDEX = 1


def _env_truthy(name: str) -> bool:
    val = os.getenv(name)
    if val is None:
        return False
    val = val.strip().lower()
    if val in ("", "0", "false", "no", "off"):
        return False
    return True


def _next_dump_dir(stage: str) -> str | None:
    global _DUMP_INDEX
    base = os.getenv("MLIR_ENABLE_DUMP_DIR", "")
    if not base:
        return None
    dump_dir = f"{base}__{_DUMP_INDEX}_{stage}"
    _DUMP_INDEX += 1
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    return dump_dir


def _mlir_debug_args(stage: str) -> list[str]:
    if not _env_truthy("MLIR_ENABLE_DUMP"):
        return []
    args = [
        "--mlir-print-ir-after-all",
        "--mlir-print-ir-module-scope",
        "--mlir-disable-threading",
    ]
    dump_dir = _next_dump_dir(stage)
    if dump_dir:
        args.append(f"--mlir-print-ir-tree-dir={dump_dir}")
    return args


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_tool(name: str, env_var: str | None = None) -> str:
    if env_var:
        env_path = os.getenv(env_var)
        if env_path:
            return env_path
    which = shutil.which(name)
    if which:
        return which
    raise RuntimeError(f"Unable to locate {name}. Set {env_var} or build the tool.")


def _get_llvm_lib_dir() -> str | None:
    libdir = os.getenv("LLVM_LIBRARY_DIR")
    if libdir:
        return libdir
    bin_dir = os.getenv("LLVM_BINARY_DIR")
    if bin_dir:
        candidate = Path(bin_dir).parent / "lib"
        if candidate.exists():
            return str(candidate)
    candidate = _repo_root() / "llvm-triton/llvm-project/build/lib"
    if candidate.exists():
        return str(candidate)
    return None


def _default_target_triple() -> str:
    triple = sysconfig.get_config_var("HOST_GNU_TYPE") or sysconfig.get_config_var("BUILD_GNU_TYPE")
    if triple:
        return triple
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "darwin":
        return f"{machine}-apple-darwin"
    if system == "linux":
        return f"{machine}-unknown-linux-gnu"
    if system == "windows":
        return f"{machine}-pc-windows-msvc"
    return machine


@dataclass(frozen=True)
class CPUOptions:
    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 1
    cluster_dims: tuple = (1, 1, 1)

    arch: str = None
    enable_fp_fusion: bool = True
    backend_name: str = "cpu"
    sanitize_overflow: bool = True

    debug: bool = False
    instrumentation_mode: str = ""
    allowed_dot_input_precisions: tuple[str] = ("ieee",)
    min_dot_size: int = 1

    def hash(self):
        hash_dict = dict(self.__dict__)
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class XYZBackend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "so"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def get_target_name(self, options) -> str:
        return "cpu"

    @functools.lru_cache()
    def hash(self):
        version = 0.1
        return f"{version}-{self.target.arch}"

    def parse_options(self, options):
        args = {
            "arch": os.getenv("TRITON_CPU_ARCH", ""),
        }
        args.update(
            {k: options[k] for k in CPUOptions.__dataclass_fields__.keys() if k in options if options[k] is not None}
        )

        return CPUOptions(**args)

    def pack_metadata(self, metadata):
        shared = getattr(metadata, "shared", 0)
        cluster_dims = getattr(metadata, "cluster_dims", (1, 1, 1))
        return (
            metadata.num_warps,
            metadata.num_ctas,
            shared,
            cluster_dims[0],
            cluster_dims[1],
            cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs, rhs: (1, 1, 1)}

    def get_module_map(self) -> Dict[str, ModuleType]:
        # TODO
        return {"triton.language.extra.libdevice": None}

    def load_dialects(self, ctx):
        # TODO
        pass

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_linalg(mod, metadata, options):
        ttir_code = str(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "ttir.mlir")
            dst_path = os.path.join(tmpdir, "linalg.mlir")
            Path(src_path).write_text(ttir_code)
            cmd = [_find_tool("triton-xyz-opt")]
            cmd.extend(_mlir_debug_args("ttir_to_linalg"))
            cmd.extend(
                [
                    src_path,
                    "--triton-to-linalg=pids-to-func-args=true",
                    "-o",
                    dst_path,
                ]
            )
            subprocess.check_call(cmd)
            return Path(dst_path).read_text()

    @staticmethod
    def make_llir(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            linalg_path = os.path.join(tmpdir, "linalg.mlir")
            llvm_path = os.path.join(tmpdir, "llvm.mlir")
            llir_path = os.path.join(tmpdir, "ll.ir")
            Path(linalg_path).write_text(src)
            cmd = [_find_tool("mlir-opt")]
            cmd.extend(_mlir_debug_args("linalg_to_llvm"))
            cmd.extend(
                [
                    linalg_path,
                    "--one-shot-bufferize",
                    "--convert-linalg-to-loops",
                    "--lower-affine",
                    "--convert-scf-to-cf",
                    "--memref-expand",
                    "--expand-strided-metadata",
                    "--finalize-memref-to-llvm",
                    "--convert-to-llvm",
                    "--reconcile-unrealized-casts",
                    "-o",
                    llvm_path,
                ]
            )
            subprocess.check_call(cmd)
            subprocess.check_call(
                [
                    _find_tool("mlir-translate"),
                    llvm_path,
                    "--mlir-to-llvmir",
                    "-o",
                    llir_path,
                ]
            )
            metadata["shared"] = 0
            return Path(llir_path).read_text()

    @staticmethod
    def make_asm(src, metadata, options):
        names = re.findall(r"define void @(?!(?:barrier)\\b)([a-zA-Z_][a-zA-Z0-9_]*)", src)
        if len(names) != 1:
            raise RuntimeError(f"Expected 1 kernel function, found {names}")
        metadata["name"] = names[0]
        llvm.init_targets()
        triple = _default_target_triple()
        proc = options.arch or ""
        flags: list[str] = []
        return llvm.translate_to_asm(src, triple, proc, "", flags, options.enable_fp_fusion, False)

    @staticmethod
    def make_library(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            Path(asm_path).write_text(src)
            lib_dirs = []
            libs = []
            ccflags = []
            llvm_lib_dir = _get_llvm_lib_dir()
            if llvm_lib_dir:
                lib_dirs.append(llvm_lib_dir)
                libs.extend(["mlir_runner_utils", "mlir_c_runner_utils"])
                for lib_dir in lib_dirs:
                    ccflags.extend(["-Wl,-rpath", lib_dir])
            so = _build("kernel", asm_path, tmpdir, lib_dirs, [], libs, ccflags)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options, language):  # ty:ignore[invalid-method-override]
        if language == Language.GLUON:
            raise Exception("GLUON is not supported")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)
