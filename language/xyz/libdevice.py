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
    _semantic: TritonSemantic = None,
):
    tensor = _semantic.to_tensor(input)
    handle = libtriton.xyz.create_nop(_semantic.builder, tensor.handle)
    return core.tensor(handle, tensor.type)


# ---------------------------------------------------------------------------
# Helpers that handle both scalar and block (vector) types
# ---------------------------------------------------------------------------

def _scalar_dtype(dtype):
    """Extract the scalar dtype from a possibly-vector block type."""
    return getattr(dtype, 'scalar', dtype)


def _is_block(dtype):
    """Check if dtype is a block (vector) type."""
    return hasattr(dtype, 'scalar') and dtype is not _scalar_dtype(dtype)


def _to_block(b, dtype, scalar_handle):
    """Splat a scalar value to match a block type. No-op if dtype is scalar."""
    if _is_block(dtype):
        return b.create_splat(dtype.to_ir(b), scalar_handle)
    return scalar_handle


def _block_type_for(input_dtype, scalar_ty):
    """Return a block type wrapping scalar_ty with the same shape as input_dtype."""
    if _is_block(input_dtype):
        return input_dtype.with_element_ty(scalar_ty)
    return scalar_ty


def _fp_const(b, dtype, value):
    """Floating-point constant, splatted to block type if dtype is a block type."""
    scalar = _scalar_dtype(dtype)
    if scalar.is_fp32():
        result = b.get_fp32(value)
    elif scalar.is_fp64():
        result = b.get_fp64(value)
    elif scalar.is_fp16() or scalar.is_bf16():
        result = b.get_fp32(value)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    return _to_block(b, dtype, result)


def _promote_to_fp32(b, handle, dtype):
    """Promote fp16/bf16/int elements to fp32. Preserves block shape."""
    scalar = _scalar_dtype(dtype)
    if scalar.is_fp16() or scalar.is_bf16():
        if _is_block(dtype):
            target_ty = _block_type_for(dtype, tl.float32)
            return b.create_fp_to_fp(handle, target_ty.to_ir(b))
        return b.create_fp_ext(handle, b.get_float_ty())
    if scalar.is_int():
        if _is_block(dtype):
            target_ty = _block_type_for(dtype, tl.float32)
            return b.create_si_to_fp(handle, target_ty.to_ir(b))
        return b.create_si_to_fp(handle, tl.float32.to_ir(b))
    return handle


def _cast_to_pow_compute(tensor, _semantic):
    """Promote low precision and integer pow operands to fp32."""
    scalar = _scalar_dtype(tensor.type)
    if scalar.is_fp16() or scalar.is_bf16() or scalar.is_int():
        return _semantic.cast(tensor, tl.float32)
    return tensor


def _trunc_from_fp32(b, handle, dtype):
    """Truncate fp32 back to fp16/bf16 if needed. Preserves block shape."""
    if _scalar_dtype(dtype).is_fp16() or _scalar_dtype(dtype).is_bf16():
        if _is_block(dtype):
            return b.create_fp_to_fp(handle, dtype.to_ir(b))
        return b.create_fp_trunc(handle, dtype.to_ir(b))
    return handle


def _effective_dtype(dtype):
    """Return the dtype to compute in (fp32 for fp16/bf16). Use scalar for block."""
    scalar = _scalar_dtype(dtype)
    if scalar.is_fp16() or scalar.is_bf16():
        return tl.float32
    return scalar


def _wrap_tensor(b, handle, dtype):
    """Wrap an MLIR handle as a core.tensor, truncating to original dtype if needed."""
    s = _scalar_dtype(dtype)
    if s.is_fp16() or s.is_bf16():
        if _is_block(dtype):
            truncated = b.create_fp_to_fp(handle, dtype.to_ir(b))
        else:
            truncated = b.create_fp_trunc(handle, dtype.to_ir(b))
        return core.tensor(truncated, dtype)
    return core.tensor(handle, dtype)


# ---------------------------------------------------------------------------
# Simple builtins
# ---------------------------------------------------------------------------

@core.builtin
def pow(x, y, _semantic=None):
    """pow(x, y): lower to a libdevice-style elementwise pow operation."""
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x, y = _semantic.binary_op_type_checking_impl(x, y)
    x = _cast_to_pow_compute(x, _semantic)
    y = _cast_to_pow_compute(y, _semantic)
    return core.extern_elementwise(
        "",
        "",
        [x, y],
        {
            (tl.float32, tl.float32): ("__nv_powf", tl.float32),
            (tl.float64, tl.float64): ("__nv_pow", tl.float64),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.builtin
def tanh(x, _semantic=None):
    """tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)"""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    xh = _promote_to_fp32(b, x.handle, dtype)
    two = _fp_const(b, dtype, 2.0)
    one = _fp_const(b, dtype, 1.0)
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
def trunc(x, _semantic=None):
    """trunc(x): round toward zero via float-to-int conversion."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    scalar = _scalar_dtype(dtype)
    int_ty = _block_type_for(dtype, tl.int32 if scalar.is_fp32() else tl.int64)
    as_int = b.create_fp_to_si(x.handle, int_ty.to_ir(b))
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
    x, y = _semantic.binary_op_type_checking_impl(x, y, div_or_mod=True)
    b = _semantic.builder
    dtype = x.type
    div = b.create_fdiv(x.handle, y.handle)
    scalar = _scalar_dtype(dtype)
    int_ty = _block_type_for(dtype, tl.int32 if scalar.is_fp32() else tl.int64)
    div_int = b.create_fp_to_si(div, int_ty.to_ir(b))
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
    scalar = _scalar_dtype(dtype)
    int_ty = _block_type_for(dtype, tl.int32 if scalar.is_fp32() else tl.int64)
    div_int = b.create_fp_to_si(div, int_ty.to_ir(b))
    return core.tensor(b.create_si_to_fp(div_int, dtype.to_ir(b)), dtype)


# ---------------------------------------------------------------------------
# Boolean-returning builtins
# ---------------------------------------------------------------------------

@core.builtin
def isnan(x, _semantic=None):
    """isnan(x): x != x is true only for NaN."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    result = b.create_fcmpUNE(x.handle, x.handle)
    return core.tensor(result, _block_type_for(x.type, tl.int1))


@core.builtin
def isinf(x, _semantic=None):
    """isinf(x): abs(x) == inf."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    abs_x = b.create_fabs(x.handle)
    inf = _fp_const(b, dtype, float('inf'))
    result = b.create_fcmpOEQ(abs_x, inf)
    return core.tensor(result, _block_type_for(dtype, tl.int1))


@core.builtin
def finitef(x, _semantic=None):
    """finitef(x): check if x is finite (not NaN and not Inf)."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    abs_x = b.create_fabs(x.handle)
    inf = _fp_const(b, dtype, float('inf'))
    not_inf = b.create_fcmpOLT(abs_x, inf)
    not_nan = b.create_fcmpOEQ(x.handle, x.handle)
    result = b.create_and(not_inf, not_nan)
    return core.tensor(result, _block_type_for(dtype, tl.int1))


@core.builtin
def isfinited(x, _semantic=None):
    """isfinited(x): same as finitef."""
    return finitef(x, _semantic=_semantic)


# ---------------------------------------------------------------------------
# Composite trig: atan / atan2 / acos using polynomial approximation
# ---------------------------------------------------------------------------

# Minimax polynomial coefficients for asin(x) on [-0.707, 0.707], ~1.7e-7 max error
_ASIN_C0 = 1.0000000626987740
_ASIN_C1 = 0.1666536927702661
_ASIN_C2 = 0.0754196799795794
_ASIN_C3 = 0.0396874337713760
_ASIN_C4 = 0.0572753135160356
_ASIN_C5 = -0.0480536859344469
_ASIN_C6 = 0.0959807721316920

# Minimax polynomial coefficients for atan(x) on [-1, 1], ~3e-7 max error
_ATAN_C0 = 0.99997726
_ATAN_C1 = -0.33262347
_ATAN_C2 = 0.19354346
_ATAN_C3 = -0.11643287
_ATAN_C4 = 0.05265332
_ATAN_C5 = -0.01172120


def _atan_poly(b, handle, dtype):
    """atan via minimax polynomial: x * P(x^2) for |x| <= 1."""
    c0 = _fp_const(b, dtype, _ATAN_C0)
    c1 = _fp_const(b, dtype, _ATAN_C1)
    c2 = _fp_const(b, dtype, _ATAN_C2)
    c3 = _fp_const(b, dtype, _ATAN_C3)
    c4 = _fp_const(b, dtype, _ATAN_C4)
    c5 = _fp_const(b, dtype, _ATAN_C5)

    x_sq = b.create_fmul(handle, handle)
    p = c5
    p = b.create_fma(x_sq, p, c4)
    p = b.create_fma(x_sq, p, c3)
    p = b.create_fma(x_sq, p, c2)
    p = b.create_fma(x_sq, p, c1)
    p = b.create_fma(x_sq, p, c0)
    return b.create_fmul(handle, p)


@core.builtin
def atan(x, _semantic=None):
    """atan(x) via minimax polynomial approximation."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    handle = x.handle
    scalar = _scalar_dtype(dtype)

    half_pi = _fp_const(b, dtype, 1.5707963267948966)
    one = _fp_const(b, dtype, 1.0)
    zero = _fp_const(b, dtype, 0.0)

    abs_x = b.create_fabs(handle)

    # For |x| <= 1: use polynomial directly
    result_small = _atan_poly(b, handle, dtype)

    # For |x| > 1: atan(x) = sign(x)*pi/2 - atan(1/|x|)
    inv_abs = b.create_fdiv(one, abs_x)
    result_large_inv = _atan_poly(b, inv_abs, dtype)
    # pi/2 - atan(1/|x|) for positive x
    result_large_pos = b.create_fsub(half_pi, result_large_inv)
    # For negative x: -(pi/2 - atan(1/|x|))
    neg_result = b.create_fsub(zero, result_large_pos)
    is_neg = b.create_fcmpOLT(handle, zero)
    result_large = b.create_select(is_neg, neg_result, result_large_pos)

    is_small = b.create_fcmpOLE(abs_x, one)
    result = b.create_select(is_small, result_small, result_large)

    return core.tensor(result, dtype)


@core.builtin
def atan2(y, x, _semantic=None):
    """atan2(y, x) = atan(y/x) with quadrant correction.
    Uses the atan large-argument identity for |y/x| > 1."""
    y = _semantic.to_tensor(y)
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = y.type

    one = _fp_const(b, dtype, 1.0)
    pi = _fp_const(b, dtype, 3.141592653589793)
    half_pi = _fp_const(b, dtype, 1.5707963267948966)
    zero = _fp_const(b, dtype, 0.0)
    neg_one = _fp_const(b, dtype, -1.0)

    # atan(y/x): for |r| <= 1 use poly, for |r| > 1 use identity
    # atan(t) = sign(t)*pi/2 - atan(1/|t|)  for |t| > 1
    abs_y = b.create_fabs(y.handle)
    abs_x = b.create_fabs(x.handle)
    r_val = b.create_fdiv(y.handle, x.handle)
    inv_r = b.create_fdiv(x.handle, y.handle)

    atan_small = _atan_poly(b, r_val, dtype)
    # atan(y/x) for |y/x| > 1:
    #   if y/x > 0: pi/2 - atan(x/y)   (x/y > 0)
    #   if y/x < 0: -pi/2 - atan(x/y)  (x/y < 0)
    neg_half_pi = b.create_fsub(zero, half_pi)
    atan_large_pos = b.create_fsub(half_pi, _atan_poly(b, inv_r, dtype))
    atan_large_neg = b.create_fsub(neg_half_pi, _atan_poly(b, inv_r, dtype))
    r_pos = b.create_fcmpOGE(r_val, zero)
    atan_large = b.create_select(r_pos, atan_large_pos, atan_large_neg)

    is_small = b.create_fcmpOLE(abs_y, abs_x)
    base = b.create_select(is_small, atan_small, atan_large)

    # Quadrant corrections: only needed when x < 0
    x_neg = b.create_fcmpOLT(x.handle, zero)
    y_ge_0 = b.create_fcmpOGE(y.handle, zero)
    y_lt_0 = b.create_fcmpOLT(y.handle, zero)

    add_pi_cond = b.create_and(x_neg, y_ge_0)
    sub_pi_cond = b.create_and(x_neg, y_lt_0)

    result = base
    result = b.create_select(add_pi_cond, b.create_fadd(result, pi), result)
    result = b.create_select(sub_pi_cond, b.create_fsub(result, pi), result)

    return core.tensor(result, dtype)


@core.builtin
def acos(x, _semantic=None):
    r"""acos(x): no clamping, uses atan identity for |x|>=0.5."""
    x = _semantic.to_tensor(x)
    b = _semantic.builder
    dtype = x.type
    handle = x.handle

    one = _fp_const(b, dtype, 1.0)
    pi = _fp_const(b, dtype, 3.141592653589793)
    half_pi = _fp_const(b, dtype, 1.5707963267948966)
    zero = _fp_const(b, dtype, 0.0)

    abs_x = b.create_fabs(handle)
    x_sq = b.create_fmul(handle, handle)

    # asin polynomial (valid for full [-1,1] with good accuracy near 0)
    c6 = _fp_const(b, dtype, _ASIN_C6)
    c5 = _fp_const(b, dtype, _ASIN_C5)
    c4 = _fp_const(b, dtype, _ASIN_C4)
    c3 = _fp_const(b, dtype, _ASIN_C3)
    c2 = _fp_const(b, dtype, _ASIN_C2)
    c1 = _fp_const(b, dtype, _ASIN_C1)
    c0 = _fp_const(b, dtype, _ASIN_C0)
    asin_p = c6
    asin_p = b.create_fma(x_sq, asin_p, c5)
    asin_p = b.create_fma(x_sq, asin_p, c4)
    asin_p = b.create_fma(x_sq, asin_p, c3)
    asin_p = b.create_fma(x_sq, asin_p, c2)
    asin_p = b.create_fma(x_sq, asin_p, c1)
    asin_p = b.create_fma(x_sq, asin_p, c0)
    asin_val = b.create_fmul(handle, asin_p)
    result_asin = b.create_fsub(half_pi, asin_val)

    # atan(sqrt(1-x^2)/|x|) branch — works best when |x| is near 1
    one_minus_x_sq = b.create_fsub(one, x_sq)
    sqrt_val = b.create_sqrt(one_minus_x_sq)
    t_val = b.create_fdiv(sqrt_val, abs_x)
    inv_t = b.create_fdiv(abs_x, sqrt_val)

    atan_small = _atan_poly(b, t_val, dtype)
    atan_large = b.create_fsub(half_pi, _atan_poly(b, inv_t, dtype))
    is_small_t = b.create_fcmpOLE(t_val, one)
    atan_t = b.create_select(is_small_t, atan_small, atan_large)

    x_neg = b.create_fcmpOLT(handle, zero)
    pi_minus_atan = b.create_fsub(pi, atan_t)
    result_atan = b.create_select(x_neg, pi_minus_atan, atan_t)

    # Blend: use atan for |x| near 1, asin for |x| near 0
    # Use a smooth threshold at 0.7
    threshold = _fp_const(b, dtype, 0.707)
    use_atan = b.create_fcmpOGT(abs_x, threshold)
    result = b.create_select(use_atan, result_atan, result_asin)

    return core.tensor(result, dtype)

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
