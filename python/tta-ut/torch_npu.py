import sys
import numpy as np
import torch

import triton  # noqa
import triton.language as tl  # noqa
from triton.backends.xyz.driver import XYZDriver  # noqa

# set device

triton.runtime.driver.set_active(XYZDriver())

DEVICE = "cpu"
torch.cpu.set_device(DEVICE)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# fake apis

torch.Tensor.npu = torch.Tensor.cpu  # ty:ignore


def _wrap_api(func):
    def wrapper(*args, **kwargs):
        if "device" in kwargs and "npu" in str(kwargs["device"]):
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)

    return wrapper


for name in ["full", "empty", "rand", "randn", "randint", "zeros", "arange"]:
    if hasattr(torch, name):
        setattr(torch, name, _wrap_api(getattr(torch, name)))

if hasattr(tl, "randint4x"):
    randint4x = getattr(tl, "randint4x")
    if not getattr(randint4x, "_ttxt_randint4x_compat", False):
        _randint4x_orig = randint4x

        @triton.jit
        def _randint4x_compat(seed, offset, n_rounds: tl.constexpr = 10):  # ty:ignore
            ret, _, _, _ = _randint4x_orig(seed, offset, n_rounds)
            return ret

        _randint4x_compat._ttxt_randint4x_compat = True  # ty:ignore
        setattr(tl, "randint4x", _randint4x_compat)

sys.modules["triton.language.extra.cann"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.cann.libdevice"] = triton.language.extra.xyz.libdevice  # ty:ignore
