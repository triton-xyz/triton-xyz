import functools
import importlib
from contextvars import ContextVar
from typing import Any, Dict

import triton
import triton.profiler as proton_api
from triton import knobs
from triton._C.libproton import proton as libproton
from triton._C.libtriton import getenv
from triton._C.libtriton import ir as triton_ir
from triton._C.libtriton import proton as triton_proton
from triton.profiler.flags import flags
from triton.profiler.hooks.hook import Hook

kernel_name = ContextVar("kernel_name", default=None)
kernel_scope_id = ContextVar("kernel_scope_id", default=None)
PUBLIC_CPU_BACKEND = "cpu"
INTERNAL_CPU_BACKEND = "cupti"
proton_profile = importlib.import_module("triton.profiler.profile")


def _patch_cpu_backend() -> None:
    if getattr(proton_profile.start, "_triton_xyz_cpu_patched", False):
        return

    original_check_env = proton_profile._check_env
    original_select_backend = proton_profile._select_backend
    original_start = proton_profile.start

    def cpu_aware_check_env(backend: str) -> None:
        target_backend = triton.runtime.driver.active.get_current_target().backend
        if backend == INTERNAL_CPU_BACKEND and target_backend == PUBLIC_CPU_BACKEND:
            for attr, desc in knobs.proton.knob_descriptors.items():
                key = desc.key
                if getenv(key, None) is not None:
                    continue
                val = getattr(knobs.proton, attr)
                if val is None:
                    continue
                if env_val := knobs.toenv(val):
                    knobs.setenv(key, env_val[0])
            return
        original_check_env(backend)

    @functools.wraps(original_select_backend)
    def cpu_aware_select_backend() -> str:
        target_backend = triton.runtime.driver.active.get_current_target().backend
        if target_backend == PUBLIC_CPU_BACKEND:
            return PUBLIC_CPU_BACKEND
        return original_select_backend()

    @functools.wraps(original_start)
    def cpu_aware_start(
        name=None,
        *,
        context="shadow",
        data="tree",
        backend=None,
        mode=None,
        hook=None,
    ):
        if backend is None:
            backend = cpu_aware_select_backend()
        internal_backend = INTERNAL_CPU_BACKEND if backend == PUBLIC_CPU_BACKEND else backend
        return original_start(
            name=name,
            context=context,
            data=data,
            backend=internal_backend,
            mode=mode,
            hook=hook,
        )

    cpu_aware_start._triton_xyz_cpu_patched = True
    proton_profile._check_env = cpu_aware_check_env
    proton_profile._select_backend = cpu_aware_select_backend
    proton_profile.start = cpu_aware_start
    proton_api.start = cpu_aware_start


_patch_cpu_backend()


class CPUInstrumentationHook(Hook):
    priority: int = 0

    def __init__(self):
        self.instrumentation_mode = "cpu"

    def activate(self) -> None:
        flags.instrumentation_on = True
        knobs.compilation.instrumentation_mode = self.instrumentation_mode

    def deactivate(self) -> None:
        flags.instrumentation_on = False
        knobs.compilation.instrumentation_mode = ""

    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        del module
        del hash
        if not function:
            return

        ir_path = next((path for key, path in metadata_group.items() if key.endswith("ttir")), None)
        if ir_path is None:
            raise RuntimeError(f"TTIR path not found in metadata for function {function}")

        context = triton_ir.context()
        triton_ir.load_dialects(context)
        triton_proton.load_dialects(context)
        module = triton_ir.parse_mlir_module(ir_path, context)
        module.context = context

        scope_id_names = triton_proton.get_scope_id_names(module)
        scope_id_parents = triton_proton.get_scope_id_parents(module)
        libproton.init_cpu_instrumentation_metadata(function, name, scope_id_names, scope_id_parents)

    def enter(self, metadata) -> None:
        func = metadata.data.get("function")
        name = metadata.data.get("name")
        if func is None or name is None:
            return

        kernel_name.set(name)
        kernel_scope_id.set(libproton.record_scope())
        libproton.enter_cpu_instrumentation(func)
        libproton.enter_scope(kernel_scope_id.get(), name)

    def exit(self, metadata) -> None:
        func = metadata.data.get("function")
        name = kernel_name.get()
        scope_id = kernel_scope_id.get()
        if scope_id is not None and name is not None:
            libproton.exit_scope(scope_id, name)
        if func is not None:
            libproton.exit_cpu_instrumentation(func)
