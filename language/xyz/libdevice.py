from triton.language import math as _math
from triton._C import libtriton
from triton.language import core
from triton.language.semantic import TritonSemantic
import triton.language as tl

# Re-export standard math builtins that FlagGems and other clients expect
# from tl_extra_shim (which resolves to this module for the xyz backend).
_MATH_EXPORTS = [
    "abs", "ceil", "cos", "div_rn", "erf", "exp", "exp2", "fdiv",
    "floor", "fma", "log", "log2", "rsqrt", "sin", "sqrt", "sqrt_rn",
    "umulhi",
]
for _name in _MATH_EXPORTS:
    if hasattr(_math, _name):
        globals()[_name] = getattr(_math, _name)


# ---------------------------------------------------------------------------
# xyz-specific builtins
# ---------------------------------------------------------------------------

@core.builtin
def nop(
    input,
    _semantic: TritonSemantic = None,  # ty:ignore
):
    tensor = _semantic.to_tensor(input)
    handle = libtriton.xyz.create_nop(_semantic.builder, tensor.handle)
    return core.tensor(handle, tensor.type)


# ---------------------------------------------------------------------------
# Composite math functions implemented using existing builder operations
# ---------------------------------------------------------------------------

def _fp_const(b, dtype, value):
    """Helper to create a floating-point constant of the right type."""
    if dtype.is_fp32():
        return b.get_fp32(value)
    elif dtype.is_fp64():
        return b.get_fp64(value)
    elif dtype.is_fp16() or dtype.is_bf16():
        return b.get_fp32(value)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def _promote_to_fp32(b, handle, dtype):
    """Promote fp16/bf16 to fp32."""
    if dtype.is_fp16() or dtype.is_bf16():
        return b.create_fp_ext(handle, b.get_float_ty())
    return handle


def _trunc_from_fp32(b, handle, dtype):
    """Truncate fp32 back to fp16/bf16 if needed."""
    if dtype.is_fp16() or dtype.is_bf16():
        return b.create_fp_trunc(handle, dtype.to_ir(b))
    return handle


def _effective_dtype(dtype):
    """Return the dtype to compute in (fp32 for fp16/bf16)."""
    if dtype.is_fp16() or dtype.is_bf16():
        return tl.float32
    return dtype


def _wrap_tensor(b, handle, dtype):
    """Wrap an MLIR handle as a core.tensor, truncating if needed."""
    if dtype.is_fp16() or dtype.is_bf16():
        truncated = b.create_fp_trunc(handle, dtype.to_ir(b))
        return core.tensor(truncated, dtype)
    return core.tensor(handle, dtype)


@core.builtin
def pow(x, y, _semantic=None):
    """pow(x, y) = exp(y * log(x))"""
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    b = _semantic.builder
    log_x = b.create_log(x.handle)
    y_log_x = b.create_fmul(y.handle, log_x)
    return core.tensor(b.create_exp(y_log_x), x.type)


@core.builtin
def tanh(x, _semantic=None):
    """tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)"""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    edtype = _effective_dtype(dtype)
    xh = _promote_to_fp32(b, x.handle, dtype)
    two = _fp_const(b, edtype, 2.0)
    one = _fp_const(b, edtype, 1.0)
    two_x = b.create_fmul(two, xh)
    exp_2x = b.create_exp(two_x)
    num = b.create_fsub(exp_2x, one)
    den = b.create_fadd(exp_2x, one)
    result = b.create_fdiv(num, den)
    return _wrap_tensor(b, result, dtype)


@core.builtin
def tan(x, _semantic=None):
    """tan(x) = sin(x) / cos(x)"""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    sin_x = b.create_sin(x.handle)
    cos_x = b.create_cos(x.handle)
    return core.tensor(b.create_fdiv(sin_x, cos_x), x.type)


@core.builtin
def acos(x, _semantic=None):
    """acos(x) = atan2(sqrt(1 - x^2), x)"""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    one = _fp_const(b, dtype, 1.0)
    x_sq = b.create_fmul(x.handle, x.handle)
    one_minus_x_sq = b.create_fsub(one, x_sq)
    sqrt_val = b.create_sqrt(one_minus_x_sq)
    return core.tensor(b.create_atan2(sqrt_val, x.handle), dtype)


@core.builtin
def atan(x, _semantic=None):
    """atan(x) = atan2(x, 1)"""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    one = _fp_const(b, dtype, 1.0)
    return core.tensor(b.create_atan2(x.handle, one), dtype)


@core.builtin
def atan2(y, x, _semantic=None):
    """atan2(y, x)"""
    y = _semantic.to_tensor(y)
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    return core.tensor(b.create_atan2(y.handle, x.handle), y.type)


@core.builtin
def trunc(x, _semantic=None):
    """trunc(x): round toward zero via float-to-int conversion."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    int_ty = b.get_int32_ty() if dtype.is_fp32() else b.get_int64_ty()
    as_int = b.create_fp_to_si(x.handle, int_ty)
    return core.tensor(b.create_si_to_fp(as_int, dtype.to_ir(b)), dtype)


@core.builtin
def rint(x, _semantic=None):
    """rint(x): round to nearest integer, ties to even."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    half = _fp_const(b, dtype, 0.5)
    x_plus_half = b.create_fadd(x.handle, half)
    return core.tensor(b.create_floor(x_plus_half), dtype)


@core.builtin
def fmod(x, y, _semantic=None):
    """fmod(x, y) = x - trunc(x/y) * y"""
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    b = _semantic.builder
    dtype = x.type
    div = b.create_fdiv(x.handle, y.handle)
    int_ty = b.get_int32_ty() if dtype.is_fp32() else b.get_int64_ty()
    div_int = b.create_fp_to_si(div, int_ty)
    div_trunc = b.create_si_to_fp(div_int, dtype.to_ir(b))
    prod = b.create_fmul(div_trunc, y.handle)
    return core.tensor(b.create_fsub(x.handle, prod), dtype)


@core.builtin
def div_rz(x, y, _semantic=None):
    """div_rz: division rounding toward zero."""
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    b = _semantic.builder
    dtype = x.type
    div = b.create_fdiv(x.handle, y.handle)
    int_ty = b.get_int32_ty() if dtype.is_fp32() else b.get_int64_ty()
    div_int = b.create_fp_to_si(div, int_ty)
    return core.tensor(b.create_si_to_fp(div_int, dtype.to_ir(b)), dtype)


@core.builtin
def isnan(x, _semantic=None):
    """isnan(x): x != x is true only for NaN."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    return core.tensor(b.create_fcmpUNE(x.handle, x.handle), tl.int1)


@core.builtin
def isinf(x, _semantic=None):
    """isinf(x): abs(x) == inf."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    abs_x = b.create_fabs(x.handle)
    inf = _fp_const(b, dtype, float('inf'))
    return core.tensor(b.create_fcmpOEQ(abs_x, inf), tl.int1)


@core.builtin
def finitef(x, _semantic=None):
    """finitef(x): check if x is finite (not NaN and not Inf)."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    # isfinite = (abs(x) < INF) & (x == x)
    abs_x = b.create_fabs(x.handle)
    inf = _fp_const(b, dtype, float('inf'))
    not_inf = b.create_fcmpOLT(abs_x, inf)
    not_nan = b.create_fcmpOEQ(x.handle, x.handle)
    is_finite = b.create_and(not_inf, not_nan)
    return core.tensor(is_finite, tl.int1)


@core.builtin
def isfinited(x, _semantic=None):
    """isfinited(x): same as finitef."""
    return finitef(x, _semantic=_semantic)


@core.builtin
def fast_tanh(x, _semantic=None):
    return tanh(x, _semantic=_semantic)


@core.builtin
def fast_erf(x, _semantic=None):
    return erf(x, _semantic=_semantic)


@core.builtin
def xpu_trunc_div(x, y, _semantic=None):
    return div_rz(x, y, _semantic=_semantic)


@core.builtin
def fast_gelu(x, _semantic=None):
    """fast_gelu(x) = x * sigmoid(1.702 * x)."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    coef = _fp_const(b, dtype, 1.702)
    z = b.create_fmul(coef, x.handle)
    zero = _fp_const(b, dtype, 0.0)
    neg_z = b.create_fsub(zero, z)
    exp_neg_z = b.create_exp(neg_z)
    one = _fp_const(b, dtype, 1.0)
    denom = b.create_fadd(one, exp_neg_z)
    sigmoid = b.create_fdiv(one, denom)
    return core.tensor(b.create_fmul(x.handle, sigmoid), dtype)
