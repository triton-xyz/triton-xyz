import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _safe_remainder(x, y):
    """PyTorch-style integer remainder without CPU integer trap cases."""
    min_value = 0
    if tl.constexpr(y.dtype.is_int16()):
        min_value = -32768
    elif tl.constexpr(y.dtype.is_int32()):
        min_value = -2147483648
    elif tl.constexpr(y.dtype.is_int64()):
        min_value = -9223372036854775808

    div_by_zero = y == 0
    signed_overflow = (x == min_value) & (y == -1)
    safe_y = tl.where(div_by_zero | signed_overflow, 1, y)

    r = x % safe_y
    c1 = r != 0
    c2 = (x < 0) ^ (safe_y < 0)
    return tl.where(c1 & c2, r + safe_y, r)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_tt(x, y):
    return _safe_remainder(x, y)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_ts(x, y):
    return _safe_remainder(x, y)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rem_st(x, y):
    return _safe_remainder(x, y)


def remainder(A, B):
    logger.debug("GEMS_XYZ REMAINDER")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return rem_tt(A, B)
    if isinstance(A, torch.Tensor):
        return rem_ts(A, B)
    if isinstance(B, torch.Tensor):
        return rem_st(A, B)
    return torch.tensor(A % B)


def remainder_(A, B):
    logger.debug("GEMS_XYZ REMAINDER_")
    if isinstance(B, torch.Tensor):
        return rem_tt(A, B, out0=A)
    return rem_ts(A, B, out0=A)
