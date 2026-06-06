import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import div_mode as _div_mode
from flag_gems.ops.div import div_mode_ as _div_mode_
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_extension import div_rn, div_rz, fmod, trunc

logger = logging.getLogger(__name__)


@triton.jit
def _min_value(dtype: tl.constexpr):
    value = 0
    if dtype.is_int16():
        value = -32768
    elif dtype.is_int32():
        value = -2147483648
    elif dtype.is_int64():
        value = -9223372036854775808
    return value


@triton.jit
def _safe_int_floordiv(x, y):
    min_value = _min_value(y.dtype)
    div_by_zero = y == 0
    signed_overflow = (x == min_value) & (y == -1)
    safe_y = tl.where(div_by_zero | signed_overflow, 1, y)

    r = x % safe_y
    q = x // safe_y
    c1 = r != 0
    c2 = (x < 0) ^ (safe_y < 0)
    return tl.where(c1 & c2, q - 1, q)


@triton.jit
def _safe_int_truncdiv(x, y):
    min_value = _min_value(y.dtype)
    div_by_zero = y == 0
    signed_overflow = (x == min_value) & (y == -1)
    safe_y = tl.where(div_by_zero | signed_overflow, 1, y)
    return x // safe_y


@triton.jit
def _safe_float_floordiv(x, y):
    remainder = fmod(x, y)
    imperfect = remainder != 0.0
    different_sign = (x < 0) ^ (y < 0)

    q = div_rn(x - remainder, y)
    q = tl.where(imperfect & different_sign, q - 1, q)

    floor_q = tl.math.floor(q)
    c = q - floor_q > 0.5
    floor_q = tl.where(c, floor_q + 1.0, floor_q)

    q_is_zeros = q == 0.0
    floor_q = tl.where(q_is_zeros, tl.where(different_sign, -0.0, 0.0), floor_q)

    is_div_by_zero = y == 0.0
    float_division = x / y
    return tl.where(is_div_by_zero, float_division, floor_q)


@triton.jit
def _truncdiv(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _safe_int_truncdiv(x, y)
    else:
        return trunc(div_rz(x, y))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _safe_int_floordiv(x, y)
    else:
        return _safe_float_floordiv(x, y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func(x, y):
    return _truncdiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_tensor_scalar(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _safe_int_floordiv(x, y)
    else:
        return _safe_float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_tensor_scalar(x, y):
    return _truncdiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def floor_div_func_scalar_tensor(x, y):
    if x.type.scalar.is_int() & y.type.scalar.is_int():
        return _safe_int_floordiv(x, y)
    else:
        return _safe_float_floordiv(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def trunc_div_func_scalar_tensor(x, y):
    return _truncdiv(x, y)


def floor_divide(A, B):
    logger.debug("GEMS_XYZ FLOOR_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return floor_div_func(A, B)
    if isinstance(A, torch.Tensor):
        return floor_div_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor):
        return floor_div_func_scalar_tensor(A, B)
    return torch.tensor(A // B)


def trunc_divide(A, B):
    logger.debug("GEMS_XYZ TRUNC_DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    if isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    if isinstance(B, torch.Tensor):
        return trunc_div_func_scalar_tensor(A, B)
    dtype = (
        torch.float32 if isinstance(A, float) or isinstance(B, float) else torch.int64
    )
    return torch.tensor(math.trunc(A / B), dtype=dtype)


def floor_divide_(A, B):
    logger.debug("GEMS_XYZ FLOOR_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return floor_div_func(A, B, out0=A)
    return floor_div_func_tensor_scalar(A, B, out0=A)


def trunc_divide_(A, B):
    logger.debug("GEMS_XYZ TRUNC_DIVIDE_")
    if isinstance(B, torch.Tensor):
        return trunc_div_func(A, B, out0=A)
    return trunc_div_func_tensor_scalar(A, B, out0=A)


def div_mode(A, B, rounding_mode=None):
    if rounding_mode == "trunc":
        return trunc_divide(A, B)
    if rounding_mode == "floor":
        return floor_divide(A, B)
    return _div_mode(A, B, rounding_mode)


def div_mode_(A, B, rounding_mode=None):
    if rounding_mode == "trunc":
        return trunc_divide_(A, B)
    if rounding_mode == "floor":
        return floor_divide_(A, B)
    return _div_mode_(A, B, rounding_mode)
