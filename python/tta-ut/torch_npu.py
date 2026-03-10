import sys
import types
import numpy as np
import torch

import triton  # noqa
import triton.language as tl  # noqa
import triton._utils as triton_utils
import triton.language.core as tl_core
import triton.language.semantic as triton_semantic
import triton.compiler.code_generator as triton_codegen
import triton.language.extra.cuda.libdevice as cuda_libdevice
import triton.language.extra.xyz.libdevice as xyz_libdevice
import triton.backends.xyz.driver as xyz_driver
from triton.backends.xyz.driver import XYZDriver  # noqa
from triton.runtime.jit import JITFunction, _normalize_ty

# set device

triton.runtime.driver.set_active(XYZDriver())

DEVICE = "cpu"
torch.cpu.set_device(DEVICE)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class _FakeNPUUtils:
    @staticmethod
    def set_device(device):
        return device


class _FakeNPUMatmul:
    allow_hf32 = True


npu = types.SimpleNamespace(
    utils=_FakeNPUUtils(),
    matmul=_FakeNPUMatmul(),
)


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


def _contains_fake_npu(value):
    if isinstance(value, torch.Tensor):
        return getattr(value, _FAKE_NPU_ATTR, False)
    if isinstance(value, (list, tuple, set)):
        return any(_contains_fake_npu(elem) for elem in value)
    if isinstance(value, dict):
        return any(_contains_fake_npu(elem) for elem in value.values())
    return False


def _inherits_fake_npu(args, kwargs):
    if _is_npu_device(kwargs.get("device")):
        return False
    if kwargs.get("device") is not None:
        return False
    return _contains_fake_npu(args) or _contains_fake_npu(kwargs)


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


def _wrap_tensor_method_result(method):
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if getattr(self, _FAKE_NPU_ATTR, False) or _inherits_fake_npu(args, kwargs):
            return _mark_fake_npu(result)
        return _clear_fake_npu(result)

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
    "__abs__",
    "__add__",
    "__mul__",
    "__neg__",
    "__radd__",
    "__rmul__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "clamp",
    "clone",
    "contiguous",
    "cos",
    "cosh",
    "exp",
    "log",
    "log10",
    "log1p",
    "permute",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "reshape",
    "transpose",
    "view",
]:
    if hasattr(torch.Tensor, name):
        setattr(torch.Tensor, name, _wrap_tensor_method_result(getattr(torch.Tensor, name)))


for name in [
    "arange",
    "as_tensor",
    "atan2",
    "cat",
    "clamp",
    "empty",
    "empty_like",
    "empty_strided",
    "exp",
    "full",
    "full_like",
    "log",
    "log10",
    "log1p",
    "ones",
    "ones_like",
    "pow",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "sin",
    "stack",
    "sqrt",
    "tensor",
    "where",
    "zeros",
    "zeros_like",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "cos",
    "cosh",
    "tan",
    "tanh",
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


def _patch_arange_range_power_of_two_check():
    current_impl = triton_semantic.TritonSemantic.arange
    if getattr(current_impl, "_ttx_arange_range_compat", False):
        return

    def _arange(self, start: int, end: int, *, ret_ty: tl.block_type = None):
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        extent = end - start
        shape = [extent]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)

    _arange._ttx_arange_range_compat = True
    triton_semantic.TritonSemantic.arange = _arange


_patch_arange_range_power_of_two_check()


def _patch_dot_input_precision():
    current_impl = triton_semantic.TritonSemantic._str_to_dot_input_precision
    if getattr(current_impl, "_ttx_dot_precision_compat", False):
        return

    def _str_to_dot_input_precision(self, input_precision):
        if isinstance(input_precision, str):
            normalized = input_precision.lower()
            allowed = tuple(precision.lower() for precision in self.builder.options.allowed_dot_input_precisions)
            if normalized in {"hf32", "tf32"} and normalized not in allowed and "ieee" in allowed:
                input_precision = "ieee"
        return current_impl(self, input_precision)

    _str_to_dot_input_precision._ttx_dot_precision_compat = True
    triton_semantic.TritonSemantic._str_to_dot_input_precision = _str_to_dot_input_precision


_patch_dot_input_precision()


def _patch_constexpr_global_annotations():
    current_impl = triton_codegen.CodeGenerator._is_constexpr_global
    if getattr(current_impl, "_ttx_constexpr_global_compat", False):
        return

    def _is_constexpr_global(self, name):
        absent_marker = object()
        val = self.gscope.get(name, absent_marker)
        if val is absent_marker:
            return False

        if triton_codegen._is_constexpr(val):
            return True

        annotation = self.gscope.get("__annotations__", {}).get(name)
        if annotation is not None:
            return _normalize_ty(annotation) == "constexpr"

        return False

    _is_constexpr_global._ttx_constexpr_global_compat = True
    triton_codegen.CodeGenerator._is_constexpr_global = _is_constexpr_global


_patch_constexpr_global_annotations()


_ASCEND_RUNTIME_KWARGS = {
    "enable_auto_bind_sub_block",
    "enable_mask_fallback_conversion",
    "enable_mixed_cv",
    "multibuffer",
    "optimize_dynamic_offset",
    "sync_solver",
}


def _patch_ascend_runtime_kwargs():
    current_impl = JITFunction._pack_args
    if getattr(current_impl, "_ttx_ascend_runtime_kwargs_compat", False):
        return

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        supported_option_fields = getattr(type(backend.parse_options({})), "__dataclass_fields__", {})
        supported_option_keys = set(supported_option_fields.keys())
        sigkeys = {param.name for param in self.params}
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in _ASCEND_RUNTIME_KWARGS or key in supported_option_keys or key in sigkeys
        }
        return current_impl(self, backend, filtered_kwargs, bound_args, specialization, options)

    _pack_args._ttx_ascend_runtime_kwargs_compat = True
    JITFunction._pack_args = _pack_args


_patch_ascend_runtime_kwargs()


_XYZ_LIBDEVICE_COMPAT_OPS = (
    "abs",
    "copysign",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "acosh",
    "asinh",
    "atanh",
    "log",
    "log10",
    "log1p",
    "exp",
    "exp2",
    "erf",
    "sqrt",
    "rsqrt",
    "ceil",
    "floor",
    "trunc",
    "pow",
)


@triton.jit
def _libdevice_acosh(x):  # ty:ignore
    return tl.log(x + tl.sqrt((x - 1.0) * (x + 1.0)))


@triton.jit
def _libdevice_asinh(x):  # ty:ignore
    return tl.log(x + tl.sqrt(x * x + 1.0))


@triton.jit
def _libdevice_atanh(x):  # ty:ignore
    return 0.5 * tl.log((1.0 + x) / (1.0 - x))


_XYZ_LIBDEVICE_JIT_COMPAT_OPS = {
    "acosh": _libdevice_acosh,
    "asinh": _libdevice_asinh,
    "atanh": _libdevice_atanh,
}


def _patch_xyz_libdevice():
    if getattr(xyz_libdevice, "_ttx_libdevice_compat", False):
        return

    for name, func in _XYZ_LIBDEVICE_JIT_COMPAT_OPS.items():
        setattr(xyz_libdevice, name, func)

    for name in _XYZ_LIBDEVICE_COMPAT_OPS:
        if hasattr(xyz_libdevice, name):
            continue
        if hasattr(cuda_libdevice, name):
            setattr(xyz_libdevice, name, getattr(cuda_libdevice, name))

    xyz_libdevice._ttx_libdevice_compat = True


_patch_xyz_libdevice()


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
sys.modules["triton.language.extra.cann.libdevice"] = xyz_libdevice  # ty:ignore
sys.modules["triton.language.extra.ascend"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.ascend.libdevice"] = xyz_libdevice  # ty:ignore
