from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
import triton  # noqa: E402
from triton.backends.xyz.driver import XYZDriver  # noqa: E402


def _env_is_bool(val: str) -> bool:
    return val.lower() in ("1", "true", "yes", "on", "y")


def _first_config_only(configs):
    if not isinstance(configs, (list, tuple)) or len(configs) <= 1:
        return configs
    if isinstance(configs, tuple):
        return configs[:1]
    return configs[:1]


def _patch_triton_autotune_first_config_only():
    if not _env_is_bool(os.environ.get("TRITON_XYZ_FIRST_CONFIG_ONLY", "1")):
        return
    if getattr(triton.autotune, "_triton_xyz_first_config_only", False):
        return

    original_triton_autotune = triton.autotune

    def patched_triton_autotune(
        configs,
        key,
        prune_configs_by=None,
        reset_to_zero=None,
        restore_value=None,
        pre_hook=None,
        post_hook=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        cache_results=False,
    ):
        return original_triton_autotune(
            _first_config_only(configs),
            key,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            restore_value=restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            cache_results=cache_results,
        )

    patched_triton_autotune._triton_xyz_first_config_only = True
    triton.autotune = patched_triton_autotune
    triton.runtime.autotuner.autotune = patched_triton_autotune


class _PhiloxGenerator:
    """Generator wrapper providing Philox-compatible state for CPU backends.

    Stores seed and offset as Python ints so that philox_backend_seed_offset
    can read/write state in the same format used by CUDA Philox generators,
    without depending on the mt19937 generator state format.
    """

    def __init__(self, seed=None):
        import random
        if seed is None:
            seed = random.randint(0, 2 ** 63 - 1)
        self._seed = seed
        self._offset = 0

    def initial_seed(self):
        return self._seed

    def get_state(self):
        # Return a fresh 16-byte tensor (2 x int64) mirroring Philox state.
        # Modifications to views of this tensor are picked up by set_state().
        t = torch.zeros(2, dtype=torch.int64)
        t[0] = self._seed
        t[1] = self._offset
        return t.view(torch.uint8)

    def set_state(self, state):
        # Read the (possibly modified) offset back from the byte tensor.
        # Uses .item() to extract a Python int, avoiding any torch op
        # interception by FlagGems.
        viewed = state.view(torch.int64)
        self._offset = int(viewed[1].item())


def _cpu_default_generators():
    generator = _PhiloxGenerator(
        seed=int(os.environ.get("TRITON_XYZ_SEED", "42")))
    return {
        "cpu": generator,
        0: generator,
        torch.device("cpu"): generator,
    }


def _cpu_device_name(device=None):
    return os.environ.get("TRITON_XYZ_DEVICE_NAME", "triton-xyz-cpu")


def _cpu_device_properties(device=None):
    return SimpleNamespace(
        multi_processor_count=max(1, os.cpu_count() or 1),
        max_threads_per_multi_processor=max(1, os.cpu_count() or 1),
        total_memory=1 << 40,
        l2_cache_size=40 * 1024 * 1024,
    )


def _cpu_mem_get_info(device=None):
    total = 1 << 40
    return total, total


def _cpu_device_capability(device=None):
    return (0, 0)


def _patch_xyz_libdevice_fallback():
    import triton.language.extra.xyz.libdevice as xyz_libdevice

    fallback_modules = []
    try:
        import triton.language.math as math_module

        fallback_modules.append(math_module)
    except ImportError:
        pass

    try:
        import triton.language.extra.libdevice as extra_libdevice

        fallback_modules.append(extra_libdevice)
    except ImportError:
        pass

    for fallback_module in fallback_modules:
        for name in dir(fallback_module):
            if name.startswith("_") or hasattr(xyz_libdevice, name):
                continue
            setattr(xyz_libdevice, name, getattr(fallback_module, name))


def _patch_torch_cpu_compat():
    cpu = torch.cpu
    if getattr(cpu, "_triton_xyz_compat", False):
        return

    if not hasattr(cpu, "default_generators"):
        cpu.default_generators = _cpu_default_generators()  # type: ignore[attr-defined]
    if cpu.current_device() == "cpu":
        cpu.current_device = lambda: 0  # type: ignore[attr-defined]
    if not hasattr(cpu, "get_device_name"):
        cpu.get_device_name = _cpu_device_name  # type: ignore[attr-defined]
    if not hasattr(cpu, "get_device_properties"):
        cpu.get_device_properties = _cpu_device_properties  # type: ignore[attr-defined]
    if not hasattr(cpu, "get_device_capability"):
        cpu.get_device_capability = _cpu_device_capability  # type: ignore[attr-defined]
    if not hasattr(cpu, "mem_get_info"):
        cpu.mem_get_info = _cpu_mem_get_info  # type: ignore[attr-defined]
    if not hasattr(cpu, "device"):
        cpu.device = lambda device=None: nullcontext()  # type: ignore[attr-defined]

    cpu._triton_xyz_compat = True  # type: ignore[attr-defined]


def _patch_triton_error_compat():
    errors = triton.compiler.errors
    if not hasattr(errors, "MLIRCompilationError") and hasattr(
        errors, "CompilationError"
    ):
        errors.MLIRCompilationError = errors.CompilationError  # type: ignore[attr-defined]


triton.runtime.driver.set_active(XYZDriver())
torch.cpu.set_device("cpu")

SEED = int(os.environ.get("TRITON_XYZ_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)

_patch_triton_autotune_first_config_only()
_patch_torch_cpu_compat()
_patch_xyz_libdevice_fallback()
_patch_triton_error_compat()
triton.runtime.driver.set_active(XYZDriver())
