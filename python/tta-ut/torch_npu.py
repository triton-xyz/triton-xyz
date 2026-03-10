import sys
import numpy as np
import torch

import triton  # noqa
import triton.compiler.code_generator as triton_codegen  # noqa
import triton.runtime.jit as triton_jit  # noqa
import triton._utils as triton_utils  # noqa
import triton.language as tl  # noqa
import triton.language.core as tl_core  # noqa
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


def _normalize_constexpr_subblocks(jit_fn, args, kwargs):
    param_names = tuple(jit_fn.signature.parameters.keys())
    kernel_kwargs = {name: value for name, value in kwargs.items() if name in param_names}
    bound = jit_fn.signature.bind_partial(*args, **kernel_kwargs)
    updates = {}
    for name, value in bound.arguments.items():
        if not name.endswith("_SUB") or not isinstance(value, int) or value <= 0 or _is_power_of_two(value):
            continue
        base_name = name[:-4]
        base_value = bound.arguments.get(base_name)
        if not isinstance(base_value, int) or base_value <= 0:
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
            args, kwargs = _normalize_constexpr_subblocks(self, args, kwargs)
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

triton.language.extra.xyz.libdevice = extra_libdevice  # ty:ignore[attr-defined]
sys.modules["triton.language.extra.cann"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.cann.libdevice"] = extra_libdevice
sys.modules["triton.language.extra.ascend"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.ascend.libdevice"] = extra_libdevice
