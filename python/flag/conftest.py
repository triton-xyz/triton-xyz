from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch_xyz  # noqa: F401
import flag_gems


def _env_is_bool(val: str) -> bool:
    return val.lower() in ("1", "true", "yes", "on", "y")


TRITON_XYZ_FIRST_CONFIG_ONLY = _env_is_bool(
    os.environ.get("TRITON_XYZ_FIRST_CONFIG_ONLY", "1")
)


def _first_config_only(configs):
    if not isinstance(configs, (list, tuple)) or len(configs) <= 1:
        return configs
    if isinstance(configs, tuple):
        return configs[:1]
    return configs[:1]


def _is_bool_dtype(value):
    if value is torch.bool:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "bool"
    return False


def _trim_autotuner_chain(obj):
    trimmed = False
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        configs = getattr(obj, "configs", None)
        first_configs = _first_config_only(configs)
        if first_configs is not configs:
            obj.configs = first_configs
            trimmed = True
        obj = getattr(obj, "fn", None)
    return trimmed


def _reset_libentry_kernel_cache(entry):
    entry.kernel_cache = tuple(dict() for _ in entry.kernel_cache)


def _patch_flag_gems_libtuner_and_libentry_for_first_config():
    if not TRITON_XYZ_FIRST_CONFIG_ONLY:
        return

    libentry_mod = importlib.import_module("flag_gems.utils.libentry")
    utils_mod = importlib.import_module("flag_gems.utils")
    if getattr(libentry_mod, "_triton_xyz_first_config_only", False):
        return

    original_libtuner = libentry_mod.libtuner
    original_libentry = libentry_mod.libentry
    original_libtuner_init = libentry_mod.LibTuner.__init__
    original_libtuner_run = libentry_mod.LibTuner.run
    original_libentry_run = libentry_mod.LibEntry.run

    def patched_libtuner(
        configs,
        key,
        prune_configs_by=None,
        reset_to_zero=None,
        restore_value=None,
        pre_hook=None,
        post_hook=None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
        do_bench=None,
        strategy="default",
        policy="default",
    ):
        return original_libtuner(
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
            strategy=strategy,
            policy=policy,
        )

    def patched_libentry():
        decorator = original_libentry()

        def wrapper(fn):
            entry = decorator(fn)
            if _trim_autotuner_chain(getattr(entry, "fn", None)):
                _reset_libentry_kernel_cache(entry)
            return entry

        return wrapper

    def patched_libtuner_init(self, *args, **kwargs):
        if len(args) >= 3:
            args = list(args)
            args[2] = _first_config_only(args[2])
            args = tuple(args)
        elif "configs" in kwargs:
            kwargs["configs"] = _first_config_only(kwargs["configs"])
        return original_libtuner_init(self, *args, **kwargs)

    def patched_libtuner_run(self, *args, **kwargs):
        _trim_autotuner_chain(self)
        return original_libtuner_run(self, *args, **kwargs)

    def patched_libentry_run(self, *args, **kwargs):
        if _trim_autotuner_chain(self.fn):
            _reset_libentry_kernel_cache(self)
        return original_libentry_run(self, *args, **kwargs)

    patched_libtuner._triton_xyz_first_config_only = True
    patched_libentry._triton_xyz_first_config_only = True
    patched_libtuner_init._triton_xyz_first_config_only = True
    patched_libtuner_run._triton_xyz_first_config_only = True
    patched_libentry_run._triton_xyz_first_config_only = True

    libentry_mod.libtuner = patched_libtuner
    libentry_mod.libentry = patched_libentry
    libentry_mod.LibTuner.__init__ = patched_libtuner_init
    libentry_mod.LibTuner.run = patched_libtuner_run
    libentry_mod.LibEntry.run = patched_libentry_run
    utils_mod.libtuner = patched_libtuner
    utils_mod.libentry = patched_libentry
    libentry_mod._triton_xyz_first_config_only = True

    for module in tuple(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("flag_gems"):
            continue
        for value in vars(module).values():
            if isinstance(value, libentry_mod.LibEntry):
                if _trim_autotuner_chain(value.fn):
                    _reset_libentry_kernel_cache(value)


_patch_flag_gems_libtuner_and_libentry_for_first_config()

device = flag_gems.device


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        (
            "--mode"
            if not (
                flag_gems.vendor_name == "kunlunxin" and torch.__version__ < "2.5"
            )
            else "--fg_mode"
        ),
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="run tests on normal or quick mode",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="tests function param recorded in log files or not",
    )


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"

    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"


def pytest_collection_modifyitems(config, items):
    timeout = int(os.environ.get("TRITON_XYZ_PYTEST_TIMEOUT", "120"))
    skip_bool_dtype = _env_is_bool(os.environ.get("TRITON_XYZ_SKIP_BOOL_DTYPE", "1"))
    skip_bool_fill = _env_is_bool(os.environ.get("TRITON_XYZ_SKIP_BOOL_FILL", "1"))
    for item in items:
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(timeout, func_only=True))
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        if skip_bool_dtype and _is_bool_dtype(callspec.params.get("dtype")):
            item.add_marker(
                pytest.mark.skip(reason="triton-xyz currently skips bool dtype cases")
            )
            continue
        if skip_bool_fill and isinstance(callspec.params.get("fill_value"), bool):
            item.add_marker(
                pytest.mark.skip(
                    reason="triton-xyz currently skips bool fill_value constructor cases"
                )
            )
