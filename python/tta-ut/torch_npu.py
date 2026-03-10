import sys
import numpy as np
import torch

import triton  # noqa
import triton.compiler.code_generator as triton_codegen  # noqa
import triton.runtime.jit as triton_jit  # noqa
import triton._utils as triton_utils  # noqa
import triton.language as tl  # noqa
import triton.language.core as tl_core  # noqa
import triton.language.extra.cuda.libdevice as cuda_libdevice  # noqa
import triton.language.extra.libdevice as extra_libdevice  # noqa
import triton.language.semantic as tl_semantic  # noqa
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
_orig_tensor_cpu = torch.Tensor.cpu


def _mark_fake_npu_tensor(value):
    if isinstance(value, torch.Tensor):
        setattr(value, _FAKE_NPU_ATTR, True)
    return value


def _clear_fake_npu_tensor(value):
    if isinstance(value, torch.Tensor) and hasattr(value, _FAKE_NPU_ATTR):
        delattr(value, _FAKE_NPU_ATTR)
    return value


def _is_fake_npu_tensor(value):
    return isinstance(value, torch.Tensor) and bool(getattr(value, _FAKE_NPU_ATTR, False))


def _tensor_cpu(self, *args, **kwargs):
    return _clear_fake_npu_tensor(_orig_tensor_cpu(self, *args, **kwargs))


def _tensor_npu(self, *args, **kwargs):
    return _mark_fake_npu_tensor(_orig_tensor_cpu(self, *args, **kwargs))


torch.Tensor.cpu = _tensor_cpu  # ty:ignore
torch.Tensor.npu = _tensor_npu  # ty:ignore


def _wrap_api(func):
    def wrapper(*args, **kwargs):
        if "device" in kwargs and "npu" in str(kwargs["device"]):
            kwargs["device"] = "cpu"
            return _mark_fake_npu_tensor(func(*args, **kwargs))
        return _clear_fake_npu_tensor(func(*args, **kwargs))

    return wrapper


for name in [
    "arange",
    "empty",
    "empty_like",
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
        setattr(torch, name, _wrap_api(getattr(torch, name)))


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for element in value:
            yield from _iter_tensors(element)
        return
    if isinstance(value, dict):
        for element in value.values():
            yield from _iter_tensors(element)


def _raise_on_cpu_tensor_args(args, kwargs):
    for index, arg in enumerate(args):
        if any(not _is_fake_npu_tensor(tensor) for tensor in _iter_tensors(arg)):
            raise ValueError(f"Pointer argument (at {index}) cannot be accessed from Triton (cpu tensor?)")
    for name, arg in kwargs.items():
        if any(not _is_fake_npu_tensor(tensor) for tensor in _iter_tensors(arg)):
            raise ValueError(f"Pointer argument ({name}) cannot be accessed from Triton (cpu tensor?)")


def _builder_allows_non_power_of_two_arange(builder):
    is_simt_mode = getattr(builder, "is_simt_mode", None)
    if callable(is_simt_mode):
        return not is_simt_mode()

    options = getattr(builder, "options", None)
    if options is None:
        return False
    return getattr(options, "backend_name", "") == "cpu"


def _is_power_of_two(value):
    return isinstance(value, int) and value > 0 and (value & (value - 1)) == 0


def _should_normalize_constexpr_block(name, value, bound_arguments):
    if not isinstance(value, int) or value <= 0 or _is_power_of_two(value):
        return False
    if name.endswith("_SUB"):
        base_name = name[:-4]
        base_value = bound_arguments.get(base_name)
        return isinstance(base_value, int) and base_value > 0
    return name == "BLOCK_SIZE" or name.endswith("_BLOCK_SIZE")


def _normalize_constexpr_blocks(jit_fn, args, kwargs):
    param_names = tuple(jit_fn.signature.parameters.keys())
    kernel_kwargs = {name: value for name, value in kwargs.items() if name in param_names}
    bound = jit_fn.signature.bind_partial(*args, **kernel_kwargs)
    updates = {}
    for name, value in bound.arguments.items():
        if not _should_normalize_constexpr_block(name, value, bound.arguments):
            continue
        new_value = value & -value
        if new_value <= 0 or new_value == value:
            continue
        updates[name] = new_value

    if not updates:
        return args, kwargs

    new_args = list(args)
    for index, name in enumerate(param_names[: len(new_args)]):
        if name in updates:
            new_args[index] = updates[name]

    new_kwargs = dict(kwargs)
    for name, value in updates.items():
        if name in new_kwargs:
            new_kwargs[name] = value

    return tuple(new_args), new_kwargs


_orig_jit_run = triton_jit.JITFunction.run
if not getattr(_orig_jit_run, "_ttx_normalize_subblocks_compat", False):

    def _jit_run_compat(self, *args, grid, warmup, **kwargs):
        target = triton.runtime.driver.active.get_current_target()
        if getattr(target, "backend", "") == "cpu":
            _raise_on_cpu_tensor_args(args, kwargs)
            args, kwargs = _normalize_constexpr_blocks(self, args, kwargs)
        return _orig_jit_run(self, *args, grid=grid, warmup=warmup, **kwargs)

    _jit_run_compat._ttx_normalize_subblocks_compat = True
    triton_jit.JITFunction.run = _jit_run_compat


_orig_triton_semantic_arange = tl_semantic.TritonSemantic.arange
if not getattr(_orig_triton_semantic_arange, "_ttx_non_power_of_two_compat", False):

    def _arange_compat(self, start, end, *, ret_ty=None):
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        range_ = end - start
        if not _builder_allows_non_power_of_two_arange(self.builder) and (range_ & (range_ - 1)) != 0:
            raise ValueError("arange's range must be a power of 2")
        shape = [range_]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)

    _arange_compat._ttx_non_power_of_two_compat = True
    tl_semantic.TritonSemantic.arange = _arange_compat


def _validate_block_shape_compat(shape):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]`")
        numel *= d
    if numel > triton_utils.TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"numel ({numel}) exceeds triton maximum tensor numel ({triton_utils.TRITON_MAX_TENSOR_NUMEL})"
        )
    return numel


if getattr(tl_core.validate_block_shape, "__name__", "") != "_validate_block_shape_compat":
    tl_core.validate_block_shape = _validate_block_shape_compat
    triton_utils.validate_block_shape = _validate_block_shape_compat


_orig_ast_to_ttir = triton_codegen.ast_to_ttir
if not getattr(_orig_ast_to_ttir, "_ttx_skip_verify_compat", False):

    def _ast_to_ttir_compat(fn, src, context, options, codegen_fns, module_map, module=None):
        if getattr(options, "backend_name", "") != "cpu":
            return _orig_ast_to_ttir(fn, src, context, options, codegen_fns, module_map, module=module)

        arg_types = [None] * len(fn.arg_names)
        for key, value in src.signature.items():
            index = fn.arg_names.index(key)
            arg_types[index] = triton_codegen.str_to_ty(value, None)

        def _apply_constexpr_types(argument, indices, value):
            index = indices.pop()
            if len(indices) == 0:
                if isinstance(argument, list):
                    argument[index] = triton_codegen.constexpr(value).type
                else:
                    argument.types[index] = triton_codegen.constexpr(value).type
            else:
                _apply_constexpr_types(argument[index], indices, value)

        for path, value in src.constants.items():
            _apply_constexpr_types(arg_types, list(path)[::-1], value)

        prototype = triton_codegen.ASTFunction([], arg_types, src.attrs)
        file_name, begin_line = triton_codegen.get_jit_fn_file_line(fn)
        from collections import namedtuple

        leaves = filter(lambda value: len(value) == 1, src.constants)
        constants = {fn.arg_names[index[0]]: src.constants[index] for index in leaves}
        proxy = namedtuple("SpecializationProxy", ["constants", "signature"])(constants, src.signature)
        generator = triton_codegen.CodeGenerator(
            context,
            prototype,
            gscope=fn.get_capture_scope(),
            function_name=fn.repr(proxy),
            jit_fn=fn,
            is_kernel=True,
            file_name=file_name,
            begin_line=begin_line,
            options=options,
            codegen_fns=codegen_fns,
            module_map=module_map,
            module=module,
            is_gluon=fn.is_gluon(),
        )
        generator.visit(fn.parse())
        module = generator.module
        module.context = context
        return module

    _ast_to_ttir_compat._ttx_skip_verify_compat = True
    triton_codegen.ast_to_ttir = _ast_to_ttir_compat

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

@triton.jit
def _libdevice_cyl_bessel_i0_compat(arg0):
    x = arg0.to(tl.float32)
    ax = tl.abs(x)

    y_small = ax / 3.75
    y_small = y_small * y_small
    small = 1.0 + y_small * (
        3.5156229
        + y_small
        * (
            3.0899424
            + y_small
            * (1.2067492 + y_small * (0.2659732 + y_small * (0.0360768 + y_small * 0.0045813)))
        )
    )

    safe_ax = tl.where(ax < 3.75, 3.75, ax)
    y_large = 3.75 / safe_ax
    large = tl.exp(safe_ax) / tl.sqrt(safe_ax) * (
        0.39894228
        + y_large
        * (
            0.01328592
            + y_large
            * (
                0.00225319
                + y_large
                * (
                    -0.00157565
                    + y_large
                    * (
                        0.00916281
                        + y_large
                        * (
                            -0.02057706
                            + y_large * (0.02635537 + y_large * (-0.01647633 + y_large * 0.00392377))
                        )
                    )
                )
            )
        )
    )

    return tl.where(ax < 3.75, small, large).to(arg0.dtype)

extra_libdevice.cyl_bessel_i0 = _libdevice_cyl_bessel_i0_compat
cuda_libdevice.cyl_bessel_i0 = _libdevice_cyl_bessel_i0_compat
triton.language.extra.xyz.libdevice = extra_libdevice  # ty:ignore[attr-defined]
sys.modules["triton.language.extra.cann"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.cann.libdevice"] = extra_libdevice
sys.modules["triton.language.extra.ascend"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.ascend.libdevice"] = extra_libdevice
