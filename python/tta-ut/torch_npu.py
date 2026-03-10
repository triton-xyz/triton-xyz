import sys
import numpy as np
import torch

import triton  # noqa
import triton.language as tl  # noqa
import triton._utils as triton_utils
import triton.language.core as tl_core
import triton.backends.xyz.driver as xyz_driver
from triton.backends.xyz.driver import XYZDriver  # noqa

# set device

triton.runtime.driver.set_active(XYZDriver())

DEVICE = "cpu"
torch.cpu.set_device(DEVICE)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# fake apis

_FAKE_NPU_ATTR = "_ttx_fake_npu"


def _is_npu_device(device):
    return device is not None and "npu" in str(device)


def _mark_fake_npu(value):
    if isinstance(value, torch.Tensor):
        value._ttx_fake_npu = True
    return value


def _clear_fake_npu(value):
    if isinstance(value, torch.Tensor) and hasattr(value, _FAKE_NPU_ATTR):
        value._ttx_fake_npu = False
    return value


def _inherits_fake_npu(args, kwargs):
    if _is_npu_device(kwargs.get("device")):
        return False
    if kwargs.get("device") is not None:
        return False
    for arg in args:
        if isinstance(arg, torch.Tensor) and getattr(arg, _FAKE_NPU_ATTR, False):
            return True
    return False


def _wrap_tensor_result(func):
    def wrapper(*args, **kwargs):
        requested_npu = _is_npu_device(kwargs.get("device"))
        inherited_npu = _inherits_fake_npu(args, kwargs)
        if requested_npu:
            kwargs["device"] = "cpu"
        result = func(*args, **kwargs)
        if requested_npu or inherited_npu:
            return _mark_fake_npu(result)
        return result

    return wrapper


_orig_tensor_to = torch.Tensor.to


def _tensor_to(self, *args, **kwargs):
    requested_npu = False
    if args and _is_npu_device(args[0]):
        args = ("cpu", *args[1:])
        requested_npu = True
    if _is_npu_device(kwargs.get("device")):
        kwargs["device"] = "cpu"
        requested_npu = True
    result = _orig_tensor_to(self, *args, **kwargs)
    if requested_npu or getattr(self, _FAKE_NPU_ATTR, False):
        return _mark_fake_npu(result)
    return _clear_fake_npu(result)


def _tensor_npu(self, *args, **kwargs):
    return _mark_fake_npu(_orig_tensor_to(self, "cpu", *args, **kwargs))


def _tensor_cpu(self, *args, **kwargs):
    return _clear_fake_npu(_orig_tensor_to(self, "cpu", *args, **kwargs))


torch.Tensor.npu = _tensor_npu  # ty:ignore
torch.Tensor.to = _tensor_to  # ty:ignore
torch.Tensor.cpu = _tensor_cpu  # ty:ignore


for name in [
    "arange",
    "as_tensor",
    "empty",
    "empty_like",
    "empty_strided",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "tensor",
    "zeros",
    "zeros_like",
]:
    if hasattr(torch, name):
        setattr(torch, name, _wrap_tensor_result(getattr(torch, name)))


def _relaxed_validate_block_shape(shape):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]"
            )
        numel *= d

    if numel > triton_utils.TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"numel ({numel}) exceeds triton maximum tensor numel ({triton_utils.TRITON_MAX_TENSOR_NUMEL})"
        )
    return numel


triton_utils.validate_block_shape = _relaxed_validate_block_shape
tl_core.validate_block_shape = _relaxed_validate_block_shape


_orig_build_unranked_memref = xyz_driver._build_unranked_memref


def _build_unranked_memref(arg, keepalive):
    if isinstance(arg, torch.Tensor) and not getattr(arg, _FAKE_NPU_ATTR, False):
        raise ValueError("Pointer argument cannot be accessed from Triton (cpu tensor?)")
    return _orig_build_unranked_memref(arg, keepalive)


xyz_driver._build_unranked_memref = _build_unranked_memref

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
sys.modules["triton.language.extra.ascend"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.ascend.libdevice"] = triton.language.extra.xyz.libdevice  # ty:ignore
