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

from triton import knobs
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, llvm, passes  # ty:ignore

_DUMP_INDEX = 1

MLIR_ENABLE_DUMP_DIR = os.getenv("MLIR_ENABLE_DUMP_DIR", "").strip()

if MLIR_ENABLE_DUMP_DIR:
    Path(MLIR_ENABLE_DUMP_DIR).mkdir(parents=True, exist_ok=True)

if MLIR_ENABLE_DUMP_DIR and not getattr(tempfile, "_tt_xyz_tmp_wrapped_compiler", False):
    tempfile.TemporaryDirectory = functools.partial(  # ty:ignore
        tempfile.TemporaryDirectory,
        dir=MLIR_ENABLE_DUMP_DIR,
        prefix="_tt_xyz_compiler_",
        delete=False,
    )
    tempfile._tt_xyz_tmp_wrapped_compiler = True  # ty:ignore


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("", "0", "false", "no", "off"):
        return False
    return True


def _next_dump_dir(stage: str) -> str | None:
    global _DUMP_INDEX
    base = os.getenv("MLIR_ENABLE_DUMP_DIR", "")
    if not base:
        return None
    dump_dir = f"{base}/_pass_dump_{_DUMP_INDEX}_{stage}"
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


def _launcher_symbol(name: str) -> str:
    return f"__tt_xyz_launch_{name}"


def _cxx_scalar_type(ty: str) -> str:
    mapping = {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }
    if ty not in mapping:
        raise ValueError(f"Unsupported launcher scalar type: {ty}")
    return mapping[ty]


def _wrapper_arg_specs(flat_signature: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    arg_idx = 0
    for ty in flat_signature:
        if ty.startswith("*"):
            specs.append(("int64_t", f"arg{arg_idx}_rank"))
            specs.append(("void*", f"arg{arg_idx}_desc"))
        else:
            specs.append((_cxx_scalar_type(ty), f"arg{arg_idx}"))
        arg_idx += 1
    return specs


def _flatten_signature_types(sig, output: list[str]) -> None:
    if isinstance(sig, tuple):
        for entry in sig:
            _flatten_signature_types(entry, output)
        return
    output.append(sig)


def _generate_launcher_wrapper(kernel_name: str, flat_signature: list[str], instrumentation_enabled: bool) -> str:
    arg_specs = _wrapper_arg_specs(flat_signature)
    arg_decl = ", ".join(f"{ty} {name}" for ty, name in arg_specs)
    kernel_arg_decl = (
        ", ".join(
            [
                arg_decl,
                "int32_t num_p0",
                "int32_t num_p1",
                "int32_t num_p2",
                "int32_t pid_x",
                "int32_t pid_y",
                "int32_t pid_z",
            ]
        )
        if arg_decl
        else ", ".join(
            ["int32_t num_p0", "int32_t num_p1", "int32_t num_p2", "int32_t pid_x", "int32_t pid_y", "int32_t pid_z"]
        )
    )
    launcher_arg_decl = (
        ", ".join([arg_decl, "int32_t num_p0", "int32_t num_p1", "int32_t num_p2", "int32_t requested_threads"])
        if arg_decl
        else ", ".join(["int32_t num_p0", "int32_t num_p1", "int32_t num_p2", "int32_t requested_threads"])
    )
    field_decl = "\n".join(f"  {ty} {name};" for ty, name in arg_specs)
    context_init = ", ".join([name for _, name in arg_specs] + ["num_p0", "num_p1", "num_p2"])
    kernel_arg_names = ", ".join(
        [f"ctx.{name}" for _, name in arg_specs] + ["ctx.num_p0", "ctx.num_p1", "ctx.num_p2", "pid_x", "pid_y", "pid_z"]
    )
    instrumentation_decl = ""
    launcher_decl = ""
    worker_enter = ""
    worker_exit = ""
    launch_symbol = _launcher_symbol(kernel_name)
    launcher_decl = f'extern "C" void {launch_symbol}({launcher_arg_decl});\n'
    if instrumentation_enabled:
        instrumentation_decl = (
            'extern "C" void proton_cpu_instrumentation_enter(uint64_t functionId);\n'
            'extern "C" void proton_cpu_instrumentation_exit(uint64_t functionId);\n'
        )
        worker_enter = f"  proton_cpu_instrumentation_enter(reinterpret_cast<uint64_t>(&{launch_symbol}));\n"
        worker_exit = f"  proton_cpu_instrumentation_exit(reinterpret_cast<uint64_t>(&{launch_symbol}));\n"

    return f"""#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

extern "C" void {kernel_name}({kernel_arg_decl});
{launcher_decl}{instrumentation_decl}
namespace {{

struct LaunchContext {{
{field_decl}
  int32_t num_p0;
  int32_t num_p1;
  int32_t num_p2;
}};

inline int32_t resolve_num_threads(int32_t requested_threads, int64_t total_programs) {{
  int32_t num_threads = requested_threads;
  if (num_threads <= 0) {{
    auto detected = std::thread::hardware_concurrency();
    num_threads = detected == 0 ? 1 : static_cast<int32_t>(detected);
  }}
  num_threads = std::max<int32_t>(1, num_threads);
  if (total_programs < num_threads) {{
    num_threads = static_cast<int32_t>(total_programs);
  }}
  return std::max<int32_t>(1, num_threads);
}}

inline void invoke_program(const LaunchContext &ctx, int64_t linear_pid) {{
  const int64_t programs_per_plane =
      static_cast<int64_t>(ctx.num_p0) * static_cast<int64_t>(ctx.num_p1);
  const int32_t pid_z =
      static_cast<int32_t>(linear_pid / programs_per_plane);
  const int64_t rem = linear_pid % programs_per_plane;
  const int32_t pid_y = static_cast<int32_t>(rem / ctx.num_p0);
  const int32_t pid_x = static_cast<int32_t>(rem % ctx.num_p0);
  {kernel_name}({kernel_arg_names});
}}

void launch_worker(const LaunchContext &ctx, std::atomic<int64_t> &next_program,
                   int64_t total_programs) {{
{worker_enter}  while (true) {{
    const int64_t linear_pid =
        next_program.fetch_add(1, std::memory_order_relaxed);
    if (linear_pid >= total_programs) {{
      break;
    }}
    invoke_program(ctx, linear_pid);
  }}
{worker_exit}}}

}} // namespace

extern "C" void {_launcher_symbol(kernel_name)}({launcher_arg_decl}) {{
  const int64_t total_programs = static_cast<int64_t>(num_p0) *
                                 static_cast<int64_t>(num_p1) *
                                 static_cast<int64_t>(num_p2);
  if (total_programs <= 0) {{
    return;
  }}

  const LaunchContext ctx{{{context_init}}};
  const int32_t num_threads =
      resolve_num_threads(requested_threads, total_programs);
  if (num_threads <= 1) {{
    std::atomic<int64_t> next_program{{0}};
    launch_worker(ctx, next_program, total_programs);
    return;
  }}

  std::atomic<int64_t> next_program{{0}};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(num_threads - 1));
  for (int32_t thread_idx = 1; thread_idx < num_threads; ++thread_idx) {{
    workers.emplace_back([&ctx, &next_program, total_programs]() {{
      launch_worker(ctx, next_program, total_programs);
    }});
  }}
  launch_worker(ctx, next_program, total_programs);
  for (auto &worker : workers) {{
    worker.join();
  }}
}}
"""


def _build_native_cpu_library(
    name: str,
    asm_src: str,
    wrapper_src: str,
    srcdir: str,
    library_dirs: list[str],
    libraries: list[str],
    ccflags: list[str],
) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(srcdir, f"{name}{suffix}")
    asm_obj = os.path.join(srcdir, "kernel.o")
    wrapper_obj = os.path.join(srcdir, "launcher.o")
    cxx = shutil.which("clang++")

    compile_asm_cmd = [
        cxx,
        "-c",
        asm_src,
        # "-O3",
        "-fPIC",
        "-o",
        asm_obj,
    ]
    compile_wrapper_cmd = [
        cxx,
        "-c",
        wrapper_src,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-pthread",
        "-o",
        wrapper_obj,
    ]
    link_cmd = [
        cxx,
        "-shared",
        "-fPIC",
        "-pthread",
        "-o",
        so,
        asm_obj,
        wrapper_obj,
    ]
    link_cmd += [f"-l{lib}" for lib in libraries]
    link_cmd += [f"-L{libdir}" for libdir in library_dirs]
    link_cmd.extend(ccflags)

    subprocess.check_call(compile_asm_cmd)
    subprocess.check_call(compile_wrapper_cmd)
    subprocess.check_call(link_cmd)
    return so


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
    use_tta: bool = True

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
            "use_tta": _env_truthy("TRITON_XYZ_USE_TTA", default=True),
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
        from triton.language.extra.xyz import libdevice

        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        from triton._C.libtriton import xyz as triton_xyz  # ty:ignore

        triton_xyz.load_dialects(ctx)

        instrumentation_mode = knobs.compilation.instrumentation_mode
        if not instrumentation_mode:
            return

        from triton._C.libtriton import proton as triton_proton  # ty:ignore

        triton_proton.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, options: CPUOptions):
        entry_name = mod.get_entry_func_name()
        entry_func = mod.get_function(entry_name)
        metadata["signature"] = list(mod.get_function_signature(entry_func))
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
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
    def make_linalg(mod, metadata, options: CPUOptions):
        ttir_code = str(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "ttir.mlir")
            dst_path = os.path.join(tmpdir, "linalg.mlir")
            Path(src_path).write_text(ttir_code)
            pipeline = "triton-to-linalg-tta"
            cmd = [_find_tool("triton-xyz-opt")]
            cmd.extend(_mlir_debug_args("ttir_to_linalg"))
            if options.instrumentation_mode:
                cmd.append("--proton-to-xyz")
            cmd.extend(
                [
                    src_path,
                    f"--{pipeline}=pids-to-func-args=true",
                    "-o",
                    dst_path,
                ]
            )
            subprocess.check_call(cmd)
            return Path(dst_path).read_text()

    @staticmethod
    def make_llir(src, metadata, options: CPUOptions):
        with tempfile.TemporaryDirectory() as tmpdir:
            linalg_path = os.path.join(tmpdir, "linalg.mlir")
            llvm_path = os.path.join(tmpdir, "llvm.mlir")
            llir_path = os.path.join(tmpdir, "ll.ll")
            Path(linalg_path).write_text(src)
            cmd = [_find_tool("triton-xyz-opt")]
            cmd.extend(_mlir_debug_args("xyz_to_llvm"))
            cmd.extend(
                [
                    linalg_path,
                    "--one-shot-bufferize",
                    "--convert-linalg-to-loops",
                    "--lower-affine",
                    "--convert-scf-to-cf",
                    "--memref-expand",
                    "--expand-strided-metadata",
                    "--convert-xyz-to-llvm",
                    "--reconcile-unrealized-casts",
                    "--canonicalize",
                    "--cse",
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
            metadata["shared"] = 1
            return Path(llir_path).read_text()

    @staticmethod
    def make_asm(src, metadata, options: CPUOptions):
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
    def make_library(src, metadata, options: CPUOptions):
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_path = os.path.join(tmpdir, "kernel.s")
            wrapper_path = os.path.join(tmpdir, "launcher.cpp")
            Path(asm_path).write_text(src)
            flat_signature: list[str] = []
            for sig in metadata["signature"]:
                if sig == "constexpr":
                    continue
                _flatten_signature_types(sig, flat_signature)
            wrapper_src = _generate_launcher_wrapper(
                metadata["name"], flat_signature, bool(options.instrumentation_mode)
            )
            Path(wrapper_path).write_text(wrapper_src)
            lib_dirs = []
            libs = []
            ccflags = []
            llvm_lib_dir = _get_llvm_lib_dir()
            if llvm_lib_dir:
                lib_dirs.append(llvm_lib_dir)
                libs.extend(["mlir_runner_utils", "mlir_c_runner_utils"])
                for lib_dir in lib_dirs:
                    ccflags.extend(["-Wl,-rpath", lib_dir])
            if options.instrumentation_mode:
                # TODO: rm fixed path
                proton_lib = _repo_root() / "build" / "libproton.so"
                if not proton_lib.exists():
                    raise RuntimeError(f"CPU instrumentation requires {proton_lib}")
                ccflags.extend([str(proton_lib), "-Wl,-rpath", str(proton_lib.parent)])
            so = _build_native_cpu_library("kernel", asm_path, wrapper_path, tmpdir, lib_dirs, libs, ccflags)
            with open(so, "rb") as f:
                return f.read()

    def add_stages(self, stages, options, language):  # ty:ignore
        if language == Language.GLUON:
            raise Exception("GLUON is not supported")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["linalg"] = lambda src, metadata: self.make_linalg(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["asm"] = lambda src, metadata: self.make_asm(src, metadata, options)
        stages["so"] = lambda src, metadata: self.make_library(src, metadata, options)
