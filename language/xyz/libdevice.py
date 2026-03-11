import triton
import triton.language as tl
import triton.language.core as tl_core
import triton.language.standard as tl_standard
from triton.language.extra.cuda import libdevice as cuda_libdevice


def _export_cuda_libdevice():
    exported = set()
    for name in dir(cuda_libdevice):
        if name.startswith("_"):
            continue
        value = getattr(cuda_libdevice, name)
        if callable(value):
            globals()[name] = value
            exported.add(name)
    return exported


_CUDA_LIBDEVICE_EXPORTS = _export_cuda_libdevice()


@triton.jit
def acosh(x):
    return tl.log(x + tl.sqrt((x - 1.0) * (x + 1.0)))


@triton.jit
def asinh(x):
    return tl.log(x + tl.sqrt(x * x + 1.0))


@triton.jit
def atanh(x):
    return 0.5 * tl.log((1.0 + x) / (1.0 - x))


@triton.jit
def hypot(x, y):
    abs_x = tl.abs(x)
    abs_y = tl.abs(y)
    hi = tl.maximum(abs_x, abs_y)
    lo = tl.minimum(abs_x, abs_y)
    safe_hi = tl.where(hi == 0.0, 1.0, hi)
    ratio = lo / safe_hi
    result = hi * tl.sqrt(1.0 + ratio * ratio)
    inf = float("inf")
    nan = float("nan")
    has_inf = (abs_x == inf) | (abs_y == inf)
    has_nan = (x != x) | (y != y)
    result = tl.where(hi == 0.0, 0.0, result)
    result = tl.where(has_inf, inf, result)
    result = tl.where(has_nan & (~has_inf), nan, result)
    return result


@triton.jit
def nearbyint(x):
    floor_x = tl.floor(x)
    ceil_x = tl.ceil(x)
    frac = x - floor_x
    floor_half = tl.floor(floor_x * 0.5)
    floor_is_even = floor_half + floor_half == floor_x
    ties = tl.where(floor_is_even, floor_x, ceil_x)
    rounded = tl.where(frac < 0.5, floor_x, tl.where(frac > 0.5, ceil_x, ties))
    return tl.where(x == x, rounded, x)


@triton.jit
def rint(x):
    return nearbyint(x)


@triton.jit
def nextafter(x, y):
    bitwidth: tl.constexpr = x.dtype.primitive_bitwidth
    uint_ty: tl.constexpr = tl_standard._get_int_dtype(bitwidth=bitwidth, signed=False)
    one = tl.full(x.shape, 1, uint_ty)
    bits = x.to(uint_ty, bitcast=True)
    advance = (y > x) == (x > 0)
    stepped = tl.where(advance, bits + one, bits - one)
    sign = (y < 0).to(uint_ty) << (bitwidth - 1)
    tiny = (sign | one).to(x.dtype, bitcast=True)
    moved = tl.where(x == 0, tiny, stepped.to(x.dtype, bitcast=True))
    same = x == y
    nan_mask = (x != x) | (y != y)
    return tl.where(nan_mask, x + y, tl.where(same, y, moved))


@triton.jit
def relu(x):
    return tl.maximum(x, 0.0)


@triton.jit
def signbit(x):
    bitwidth: tl.constexpr = x.dtype.primitive_bitwidth
    uint_ty: tl.constexpr = tl_standard._get_int_dtype(bitwidth=bitwidth, signed=False)
    bits = x.to(uint_ty, bitcast=True)
    sign_mask = tl.full(x.shape, 1, uint_ty) << (bitwidth - 1)
    return (bits & sign_mask) != 0


def _require_fp16_fp32_bf16(arg0, semantic):
    arg0 = semantic.to_tensor(arg0)
    scalar_ty = arg0.type.scalar
    if scalar_ty.is_fp16() or scalar_ty.is_fp32() or scalar_ty.is_bf16():
        return arg0
    raise ValueError(f"Expected dtype fp16/fp32/bf16, but got {scalar_ty}")


def _isfinite_impl(arg0, semantic):
    arg0 = _require_fp16_fp32_bf16(arg0, semantic)
    abs_arg0 = tl_core.tensor(semantic.builder.create_fabs(arg0.handle), arg0.type)
    inf = semantic.full(arg0.shape, float("inf"), arg0.type.scalar)
    return semantic.and_(semantic.equal(arg0, arg0), semantic.not_equal(abs_arg0, inf))


@tl_core.builtin
def finitef(arg0, _semantic=None):
    return _isfinite_impl(arg0, _semantic)


@tl_core.builtin
def isfinited(arg0, _semantic=None):
    return _isfinite_impl(arg0, _semantic)


@tl_core.builtin
def isnan(arg0, _semantic=None):
    arg0 = _require_fp16_fp32_bf16(arg0, _semantic)
    return _semantic.not_equal(arg0, arg0)


@triton.jit
def _approx_erf(x):
    abs_x = tl.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * abs_x)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    y = 1.0 - poly * tl.exp(-abs_x * abs_x)
    return tl.where(x < 0.0, -y, y)


@triton.jit
def erfinv(x):
    abs_x = tl.abs(x)
    sign = tl.where(x < 0.0, -1.0, 1.0)
    safe_abs_x = tl.where(abs_x < (1.0 - 1.0e-7), abs_x, 1.0 - 1.0e-7)
    log_term = tl.log(1.0 - safe_abs_x * safe_abs_x)
    w = 2.0 / (3.141592653589793 * 0.147) + 0.5 * log_term
    y = sign * tl.sqrt(tl.sqrt(w * w - log_term / 0.147) - w)
    for _ in range(3):
        y = y - (_approx_erf(y) - x) * 0.8862269254527579 * tl.exp(y * y)
    inf = float("inf")
    nan = float("nan")
    y = tl.where(abs_x > 1.0, nan, y)
    y = tl.where(x == 1.0, inf, y)
    y = tl.where(x == -1.0, -inf, y)
    return y


@triton.jit
def _lanczos_log_gamma_positive(x):
    z = x - 1.0
    acc = 0.9999999999998099
    acc = acc + 676.5203681218851 / (z + 1.0)
    acc = acc - 1259.1392167224028 / (z + 2.0)
    acc = acc + 771.3234287776531 / (z + 3.0)
    acc = acc - 176.6150291621406 / (z + 4.0)
    acc = acc + 12.507343278686905 / (z + 5.0)
    acc = acc - 0.13857109526572012 / (z + 6.0)
    acc = acc + 9.984369578019572e-6 / (z + 7.0)
    acc = acc + 1.5056327351493116e-7 / (z + 8.0)
    t = z + 7.5
    return 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(acc)


@triton.jit
def lgamma(x):
    use_reflection = x < 0.5
    base = tl.where(use_reflection, 1.0 - x, x)
    base_lgamma = _lanczos_log_gamma_positive(base)
    reflected = 1.1447298858494002 - tl.log(tl.abs(tl.sin(3.141592653589793 * x))) - base_lgamma
    result = tl.where(use_reflection, reflected, base_lgamma)
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole, float("inf"), result)


@triton.jit
def tgamma(x):
    use_reflection = x < 0.5
    base = tl.where(use_reflection, 1.0 - x, x)
    base_lgamma = _lanczos_log_gamma_positive(base)
    reflected = 3.141592653589793 / (tl.sin(3.141592653589793 * x) * tl.exp(base_lgamma))
    result = tl.where(use_reflection, reflected, tl.exp(base_lgamma))
    result = tl.where(x == 0.0, float("inf"), result)
    pole = (x < 0.0) & (x == tl.floor(x))
    return tl.where(pole, float("nan"), result)


@tl_core.builtin
def cyl_bessel_i0(arg0, _semantic=None):
    coeffs_a = [
        -4.41534164647933937950e-18,
        3.33079451882223809783e-17,
        -2.43127984654795469359e-16,
        1.71539128555513303061e-15,
        -1.16853328779934516808e-14,
        7.67618549860493561688e-14,
        -4.85644678311192946090e-13,
        2.95505266312963983461e-12,
        -1.72682629144155570723e-11,
        9.67580903537323691224e-11,
        -5.18979560163526290666e-10,
        2.65982372468238665035e-09,
        -1.30002500998624804212e-08,
        6.04699502254191894932e-08,
        -2.67079385394061173391e-07,
        1.11738753912010371815e-06,
        -4.41673835845875056359e-06,
        1.64484480707288970893e-05,
        -5.75419501008210370398e-05,
        1.88502885095841655729e-04,
        -5.76375574538582365885e-04,
        1.63947561694133579842e-03,
        -4.32430999505057594430e-03,
        1.05464603945949983183e-02,
        -2.37374148058994688156e-02,
        4.93052842396707084878e-02,
        -9.49010970480476444210e-02,
        1.71620901522208775349e-01,
        -3.04682672343198398683e-01,
        6.76795274409476084995e-01,
    ]
    coeffs_b = [
        -7.23318048787475395456e-18,
        -4.83050448594418207126e-18,
        4.46562142029675999901e-17,
        3.46122286769746109310e-17,
        -2.82762398051658348494e-16,
        -3.42548561967721913462e-16,
        1.77256013305652638360e-15,
        3.81168066935262242075e-15,
        -9.55484669882830764870e-15,
        -4.15056934728722208663e-14,
        1.54008621752140982691e-14,
        3.85277838274214270114e-13,
        7.18012445138366623367e-13,
        -1.79417853150680611778e-12,
        -1.32158118404477131188e-11,
        -3.14991652796324136454e-11,
        1.18891471078464383424e-11,
        4.94060238822496958910e-10,
        3.39623202570838634515e-09,
        2.26666899049817806459e-08,
        2.04891858946906374183e-07,
        2.89137052083475648297e-06,
        6.88975834691682398426e-05,
        3.36911647825569408990e-03,
        8.04490411014108831608e-01,
    ]

    builder = _semantic.builder
    arg0 = _semantic.to_tensor(arg0)
    abs_x = tl_core.tensor(builder.create_fabs(arg0.handle), arg0.type)
    x_a = _semantic.sub(_semantic.mul(abs_x, 0.5, True), 2.0, True)
    a_n_2 = 0.0
    a_n_1 = 0.0
    a_n = coeffs_a[0]
    for i in range(1, len(coeffs_a)):
        a_n_2 = a_n_1
        a_n_1 = a_n
        a_n = _semantic.sub(_semantic.mul(x_a, a_n_1, True), a_n_2, True)
        a_n = _semantic.add(a_n, coeffs_a[i], True)

    f_32 = _semantic.full(abs_x.shape, 32.0, abs_x.type.scalar)
    x_b = _semantic.sub(_semantic.fdiv(f_32, abs_x, True), 2.0, True)
    b_n_2 = 0.0
    b_n_1 = 0.0
    b_n = coeffs_b[0]
    for i in range(1, len(coeffs_b)):
        b_n_2 = b_n_1
        b_n_1 = b_n
        b_n = _semantic.sub(_semantic.mul(x_b, b_n_1, True), b_n_2, True)
        b_n = _semantic.add(b_n, coeffs_b[i], True)

    half_exp = _semantic.mul(tl_core.tensor(builder.create_exp(abs_x.handle), abs_x.type), 0.5, True)
    res_a = _semantic.mul(half_exp, _semantic.sub(a_n, a_n_2, True), True)
    res_b = _semantic.fdiv(
        _semantic.mul(half_exp, _semantic.sub(b_n, b_n_2, True), True),
        tl_core.tensor(builder.create_sqrt(abs_x.handle), abs_x.type),
        True,
    )
    cond = _semantic.less_equal(abs_x, 8.0)
    return _semantic.where(cond, res_a, res_b)


gamma = tgamma
_ttx_libdevice_compat = True

__all__ = sorted(
    _CUDA_LIBDEVICE_EXPORTS
    | {
        "acosh",
        "asinh",
        "atanh",
        "hypot",
        "nearbyint",
        "rint",
        "nextafter",
        "relu",
        "signbit",
        "finitef",
        "isfinited",
        "isnan",
        "erfinv",
        "gamma",
        "lgamma",
        "tgamma",
        "cyl_bessel_i0",
    }
)
