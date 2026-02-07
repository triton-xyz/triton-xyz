import ctypes
import os
import platform
import tempfile
import time

import triton

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


def _flatten_signature(sig, output):
    if isinstance(sig, tuple):
        for entry in sig:
            _flatten_signature(entry, output)
    else:
        output.append(sig)


def _flatten_arg(sig, arg, output):
    if isinstance(sig, tuple):
        if not isinstance(arg, tuple):
            raise ValueError("Tuple argument expected for tuple signature")
        for sub_sig, sub_arg in zip(sig, arg):
            _flatten_arg(sub_sig, sub_arg, output)
    else:
        output.append(arg)


def _ctype_for(ty: str):
    if ty.startswith("*"):
        return ctypes.c_void_p
    mapping = {
        "i1": ctypes.c_int8,
        "i8": ctypes.c_int8,
        "i16": ctypes.c_int16,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "u1": ctypes.c_uint8,
        "u8": ctypes.c_uint8,
        "u16": ctypes.c_uint16,
        "u32": ctypes.c_uint32,
        "u64": ctypes.c_uint64,
        "fp16": ctypes.c_float,
        "bf16": ctypes.c_float,
        "fp32": ctypes.c_float,
        "f32": ctypes.c_float,
        "fp64": ctypes.c_double,
    }
    if ty not in mapping:
        raise ValueError(f"Unsupported argument type: {ty}")
    return mapping[ty]


_MEMREF_DESC_CACHE: dict[int, type] = {}


def _ranked_memref_type(rank: int):
    desc = _MEMREF_DESC_CACHE.get(rank)
    if desc is not None:
        return desc

    class RankedMemref(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    _MEMREF_DESC_CACHE[rank] = RankedMemref
    return RankedMemref


def _build_unranked_memref(arg, keepalive):
    if not hasattr(arg, "data_ptr"):
        ptr = int(arg)
        return ctypes.c_int64(0), ctypes.c_void_p(ptr)

    rank = int(arg.dim())
    sizes = list(arg.shape)
    strides = list(arg.stride())
    offset_elems = int(arg.storage_offset()) if hasattr(arg, "storage_offset") else 0
    elem_size = int(arg.element_size()) if hasattr(arg, "element_size") else 1
    aligned_ptr = int(arg.data_ptr())
    allocated_ptr = aligned_ptr - offset_elems * elem_size
    desc_type = _ranked_memref_type(rank)
    desc = desc_type(
        ctypes.c_void_p(allocated_ptr),
        ctypes.c_void_p(aligned_ptr),
        ctypes.c_int64(offset_elems),
        (ctypes.c_int64 * rank)(*sizes),
        (ctypes.c_int64 * rank)(*strides),
    )
    keepalive.append(desc)
    return ctypes.c_int64(rank), ctypes.cast(ctypes.pointer(desc), ctypes.c_void_p)


def _to_c_arg(ty, arg):
    return ty(arg)


class Utils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Utils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
            f.write(kernel)
            f.flush()
            os.fsync(f.fileno())
            lib = ctypes.cdll.LoadLibrary(f.name)
            fn_ptr = getattr(lib, name)
            fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
            return (lib, fn_ptr_as_void_p, 1, 0, max(1, os.cpu_count() or 1))

    def get_device_properties(self, *args):
        return {
            "max_num_regs": (os.cpu_count() or 1) * 4,
            "max_shared_mem": 1024 * 1024 * 1024,
            "multiprocessor_count": os.cpu_count() or 1,
            "warpSize": 1,
        }


class XYZDeviceInterface:
    class HooksTimeAccessor:
        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:
        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None  # ty:ignore
        triton.compiler.CompiledKernel.launch_exit_hook = None  # ty:ignore

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()  # ty:ignore
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()  # ty:ignore

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return XYZDeviceInterface.HooksTimeAccessor(self)
        return XYZDeviceInterface.TimerEvent()


class Launcher(object):
    def __init__(self, src, metadata):
        flat_signature = []
        self._signature = list(src.signature.values())
        for sig in self._signature:
            if sig == "constexpr":
                continue
            _flatten_signature(sig, flat_signature)
        self._flat_sig = flat_signature
        self.argtypes = []
        for ty in self._flat_sig:
            if ty.startswith("*"):
                self.argtypes.extend([ctypes.c_int64, ctypes.c_void_p])
            else:
                self.argtypes.append(_ctype_for(ty))
        self.ctypes_fn = ctypes.CFUNCTYPE(
            None,
            *self.argtypes,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        )

    def __call__(
        self,
        gridX,
        gridY,
        gridZ,
        stream,
        function,
        packed_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ):
        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)

        cfunc = self.ctypes_fn(function)
        flat_args = []
        for sig, arg in zip(self._signature, args):
            if sig == "constexpr":
                continue
            _flatten_arg(sig, arg, flat_args)
        base_args = []
        keepalive = []
        for ty, arg in zip(self._flat_sig, flat_args):
            if ty.startswith("*"):
                rank, desc_ptr = _build_unranked_memref(arg, keepalive)
                base_args.extend([rank, desc_ptr])
            else:
                base_args.append(_to_c_arg(_ctype_for(ty), arg))

        gridX = int(gridX)
        gridY = int(gridY)
        gridZ = int(gridZ)
        num_p0, num_p1, num_p2 = gridX, gridY, gridZ
        for pid_z in range(gridZ):
            for pid_y in range(gridY):
                for pid_x in range(gridX):
                    cfunc(*base_args, num_p0, num_p1, num_p2, pid_x, pid_y, pid_z)

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)


class XYZDriver(DriverBase):
    def __init__(self):
        self.utils = Utils()
        self.launcher_cls = Launcher
        import torch
        import torch.cpu

        self.get_current_device = torch.cpu.current_device
        self.set_current_device = torch.cpu.set_device
        self.get_current_stream = torch.cpu.current_stream

    @staticmethod
    def is_active():  # ty:ignore[invalid-method-override]
        return True

    def get_device_interface(self):
        return XYZDeviceInterface()

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    def get_current_target(self):
        capability = platform.machine()
        warp_size = 1
        return GPUTarget("cpu", capability, warp_size)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")

    def clear_cache(self, cache):
        cache.zero_()
