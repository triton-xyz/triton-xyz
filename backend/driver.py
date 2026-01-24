import os

from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget


class Utils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Utils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        pass

    def load_binary(self, name, kernel, shared_mem, device):
        pass

    def get_device_properties(self, *args):
        return {
            "max_num_regs": os.cpu_count() * 4,
            "max_shared_mem": 1024 * 1024 * 1024,
            "multiprocessor_count": os.cpu_count(),
            "warpSize": 1,
        }


class Launcher(object):
    def __init__(self, src, metadata):
        # TODO
        self.launch = lambda *args: None

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        self.launch(gridX, gridY, gridZ, stream, function, *args)


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
    def is_active():
        return True

    def get_device_interface(self):
        import torch

        return torch.cpu

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty

    def get_current_target(self):
        capability = "cpu"
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
