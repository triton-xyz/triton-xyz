import inspect
import hashlib
import numbers
import os
import subprocess
import re
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import torch

import triton  # noqa
import triton.language as tl  # noqa
import triton._utils as triton_utils
import triton.language.core as tl_core
import triton.language.math as triton_math
import triton.language.semantic as triton_semantic
import triton.language.standard as triton_standard
import triton.compiler.code_generator as triton_codegen
import triton.runtime.jit as triton_jit
import triton.language.extra.xyz.extension as xyz_extension
import triton.language.extra.xyz.libdevice as xyz_libdevice
from triton._C import libtriton as triton_libtriton
import triton.backends.xyz.driver as xyz_driver
import triton.backends.xyz.compiler as xyz_compiler
from triton.backends.xyz.driver import XYZDriver  # noqa
from triton.runtime.jit import JITFunction, _normalize_ty

# set device

triton.runtime.driver.set_active(XYZDriver())

DEVICE = "cpu"
torch.cpu.set_device(DEVICE)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class _CompatAttrsDescriptor:
    __slots__ = (
        "arg_properties",
        "property_values",
        "constant_properties",
        "divisibility_16",
        "equal_to_1",
    )

    def __init__(self, params=None, values=None):
        self.arg_properties = {}
        self.property_values = {
            "tt.divisibility": 16,
            "tt.equal_to": 1,
        }
        self.constant_properties = {"tt.equal_to"}
        self._add_common_properties(params, values)
        self._init_slots()

    def _add_common_properties(self, params, values):
        if params is None or values is None:
            return

        assert len(params) == len(values)
        self.arg_properties["tt.divisibility"] = [
            param.num
            for param, arg in zip(params, values)
            if self.is_divisible_by_16(arg)
            and not param.do_not_specialize
            and not getattr(param, "do_not_specialize_on_alignment", False)
        ]
        self.arg_properties["tt.equal_to"] = [
            param.num
            for param, arg in zip(params, values)
            if self.is_equal_to_1(arg) and not param.do_not_specialize
        ]

    def _init_slots(self):
        for prop_name, prop_val in self.property_values.items():
            values = list(self.arg_properties.get(prop_name, ()))
            setattr(self, prop_name.removeprefix("tt.") + "_" + str(prop_val), values)

    def get_fn_attrs(self):
        attrs = {}
        for prop_name, arg_set in self.arg_properties.items():
            prop_val = self.property_values[prop_name]
            for arg in arg_set:
                attrs[arg] = attrs.get(arg, []) + [(prop_name, prop_val)]
        return attrs

    def get_constants(self):
        constants = {}
        for prop_name in self.constant_properties:
            for arg in self.arg_properties.get(prop_name, ()):
                constants[arg] = self.property_values[prop_name]
        return constants

    def filter_out_constants(self):
        filtered = type(self)()
        filtered.arg_properties = {
            prop_name: list(arg_set)
            for prop_name, arg_set in self.arg_properties.items()
            if prop_name not in self.constant_properties
        }
        filtered.property_values = {
            prop_name: prop_val
            for prop_name, prop_val in self.property_values.items()
            if prop_name not in self.constant_properties
        }
        filtered.constant_properties = set()
        filtered._init_slots()
        return filtered

    def hash(self):
        key = repr(
            (
                sorted((name, tuple(values)) for name, values in self.arg_properties.items()),
                sorted(self.property_values.items()),
                sorted(self.constant_properties),
            )
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        return {
            "arg_properties": {name: list(values) for name, values in self.arg_properties.items()},
            "cls": "AttrsDescriptor",
        }

    @classmethod
    def from_dict(cls, data):
        descriptor = cls()
        descriptor.arg_properties = {
            name: list(values) for name, values in data.get("arg_properties", {}).items()
        }
        descriptor._init_slots()
        return descriptor

    @classmethod
    def from_hints(cls, hints):
        descriptor = cls()
        items = list(hints.items() if hasattr(hints, "items") else hints)
        for prop_name, prop_val in descriptor.property_values.items():
            descriptor.arg_properties[prop_name] = [index for index, value in items if value == prop_val]
        descriptor._init_slots()
        return descriptor

    @staticmethod
    def is_divisible_by_16(arg):
        if hasattr(arg, "data_ptr"):
            return arg.data_ptr() % 16 == 0
        if isinstance(arg, int):
            return arg % 16 == 0
        return arg is None

    @staticmethod
    def is_equal_to_1(arg):
        return isinstance(arg, int) and not isinstance(arg, bool) and arg == 1

    @staticmethod
    def get_property_key(value, align):
        if align and _CompatAttrsDescriptor.is_divisible_by_16(value):
            return "D"
        if _CompatAttrsDescriptor.is_equal_to_1(value):
            return "1"
        return "N"

    def __repr__(self):
        return f"AttrsDescriptor.from_dict({self.to_dict()!r})"


def _patch_attrs_descriptor():
    import triton.backends.compiler as triton_backends_compiler
    import triton.compiler as triton_compiler_pkg
    import triton.compiler.compiler as triton_compiler

    if hasattr(triton_compiler, "AttrsDescriptor"):
        return

    triton_backends_compiler.AttrsDescriptor = _CompatAttrsDescriptor
    triton_compiler.AttrsDescriptor = _CompatAttrsDescriptor
    triton_compiler_pkg.AttrsDescriptor = _CompatAttrsDescriptor

    exported = list(getattr(triton_compiler_pkg, "__all__", ()))
    if "AttrsDescriptor" not in exported:
        triton_compiler_pkg.__all__ = [*exported, "AttrsDescriptor"]


_patch_attrs_descriptor()


def _patch_xyz_make_llir():
    current_impl = xyz_compiler.XYZBackend.make_llir
    if getattr(current_impl, "_ttx_xyz_make_llir_affine_compat", False):
        return

    def _make_llir(src, metadata, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            linalg_path = os.path.join(tmpdir, "linalg.mlir")
            llvm_path = os.path.join(tmpdir, "llvm.mlir")
            llir_path = os.path.join(tmpdir, "ll.ir")
            Path(linalg_path).write_text(src)
            cmd = [xyz_compiler._find_tool("triton-xyz-opt")]
            cmd.extend(xyz_compiler._mlir_debug_args("xyz_to_llvm"))
            cmd.extend(
                [
                    linalg_path,
                    "--one-shot-bufferize",
                    "--convert-linalg-to-loops",
                    "--lower-affine",
                    "--convert-scf-to-cf",
                    "--memref-expand",
                    "--expand-strided-metadata",
                    "--convert-xyz-to-llvm",
                    "--lower-affine",
                    "--convert-arith-to-llvm",
                    "--reconcile-unrealized-casts",
                    "--canonicalize",
                    "--cse",
                    "-o",
                    llvm_path,
                ]
            )
            subprocess.check_call(cmd)
            subprocess.check_call(
                [
                    xyz_compiler._find_tool("mlir-translate"),
                    llvm_path,
                    "--mlir-to-llvmir",
                    "-o",
                    llir_path,
                ]
            )
            metadata["shared"] = 1
            return Path(llir_path).read_text()

    _make_llir._ttx_xyz_make_llir_affine_compat = True
    xyz_compiler.XYZBackend.make_llir = staticmethod(_make_llir)


_patch_xyz_make_llir()


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


class _FakeNPUDevice:

    type = "npu"
    index = 0

    def __str__(self):
        return "npu:0"

    __repr__ = __str__

    def __eq__(self, other):
        if other is None:
            return False
        other_str = str(other)
        return "npu" in other_str or other_str == "cpu"

    def __hash__(self):
        return hash(("npu", self.index))


_FAKE_NPU_DEVICE = _FakeNPUDevice()


def _is_npu_device(device):
    return device is not None and "npu" in str(device)


def _pad_fake_npu_scalar_storage(value, min_elems=128):
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim != 0 or value.numel() != 1:
        return value
    storage = value.untyped_storage()
    if storage.nbytes() >= value.element_size() * min_elems:
        return value
    backing = torch.empty((min_elems,), dtype=value.dtype, device="cpu")
    backing.zero_()
    backing[0] = value.detach()
    if value.requires_grad:
        backing.requires_grad_(True)
    return torch.as_strided(backing, size=(), stride=())


def _mark_fake_npu(value):
    if isinstance(value, torch.Tensor):
        value = _pad_fake_npu_scalar_storage(value)
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


def _normalize_fake_npu_constructor_input(value, nested=False):
    if isinstance(value, torch.Tensor):
        if getattr(value, _FAKE_NPU_ATTR, False):
            value = _tensor_cpu(value)
        if nested:
            if value.ndim == 0:
                return value.item()
            return value.detach().cpu().tolist()
        return value
    if isinstance(value, list):
        return [_normalize_fake_npu_constructor_input(elem, nested=True) for elem in value]
    if isinstance(value, tuple):
        return tuple(_normalize_fake_npu_constructor_input(elem, nested=True) for elem in value)
    if isinstance(value, set):
        return {_normalize_fake_npu_constructor_input(elem, nested=True) for elem in value}
    if isinstance(value, dict):
        return {key: _normalize_fake_npu_constructor_input(elem, nested=True) for key, elem in value.items()}
    return value


def _wrap_tensor_constructor(func):
    def wrapper(*args, **kwargs):
        requested_npu = _is_npu_device(kwargs.get("device"))
        inherited_npu = _inherits_fake_npu(args, kwargs)
        if requested_npu:
            kwargs["device"] = "cpu"
        args = tuple(_normalize_fake_npu_constructor_input(arg) for arg in args)
        kwargs = {key: _normalize_fake_npu_constructor_input(value) for key, value in kwargs.items()}
        result = func(*args, **kwargs)
        if requested_npu or inherited_npu:
            return _mark_fake_npu(result)
        return result

    return wrapper


def _promote_fake_npu_tensor_pair(args, kwargs, lhs_name, rhs_name):
    lhs = rhs = None
    if len(args) >= 2:
        lhs, rhs = args[:2]
    else:
        lhs = kwargs.get(lhs_name)
        rhs = kwargs.get(rhs_name)
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return args, kwargs
    if not (_contains_fake_npu(lhs) or _contains_fake_npu(rhs)):
        return args, kwargs
    if lhs.dtype == rhs.dtype:
        return args, kwargs
    common_dtype = torch.promote_types(lhs.dtype, rhs.dtype)
    lhs = lhs.to(common_dtype)
    rhs = rhs.to(common_dtype)
    if len(args) >= 2:
        args = (lhs, rhs, *args[2:])
    else:
        kwargs = dict(kwargs)
        kwargs[lhs_name] = lhs
        kwargs[rhs_name] = rhs
    return args, kwargs


_orig_tensor_to = torch.Tensor.to
_orig_tensor_getattribute = torch.Tensor.__getattribute__


def _tensor_getattribute(self, name):
    if name == "device":
        try:
            if _orig_tensor_getattribute(self, _FAKE_NPU_ATTR):
                return _FAKE_NPU_DEVICE
        except AttributeError:
            pass
    return _orig_tensor_getattribute(self, name)


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
    if getattr(self, _FAKE_NPU_ATTR, False):
        kwargs = dict(kwargs)
        kwargs["copy"] = True
    return _clear_fake_npu(_orig_tensor_to(self, "cpu", *args, **kwargs))


torch.Tensor.npu = _tensor_npu  # ty:ignore
torch.Tensor.__getattribute__ = _tensor_getattribute  # ty:ignore
torch.Tensor.to = _tensor_to  # ty:ignore
torch.Tensor.cpu = _tensor_cpu  # ty:ignore

for name in [
    "__abs__",
    "__add__",
    "__getitem__",
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
    "flatten",
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
    "abs",
    "arange",
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


for name in [
    "as_tensor",
    "tensor",
]:
    if hasattr(torch, name):
        setattr(torch, name, _wrap_tensor_constructor(getattr(torch, name)))


_orig_torch_isclose = torch.isclose


def _torch_isclose(*args, **kwargs):
    args, kwargs = _promote_fake_npu_tensor_pair(args, kwargs, "input", "other")
    result = _orig_torch_isclose(*args, **kwargs)
    if _inherits_fake_npu(args, kwargs):
        return _mark_fake_npu(result)
    return result


torch.isclose = _torch_isclose


_orig_torch_allclose = torch.allclose


def _torch_allclose(*args, **kwargs):
    args, kwargs = _promote_fake_npu_tensor_pair(args, kwargs, "input", "other")
    return _orig_torch_allclose(*args, **kwargs)


torch.allclose = _torch_allclose


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


def _patch_constexpr_tensor_to_tensor():
    current_impl = triton_semantic.TritonSemantic.to_tensor
    if getattr(current_impl, "_ttx_constexpr_tensor_compat", False):
        return

    def _to_tensor(self, x, check_type=True):
        if isinstance(x, tl.constexpr):
            x = x.value
        if isinstance(x, self.tensor):
            return x
        return current_impl(self, x, check_type)

    _to_tensor._ttx_constexpr_tensor_compat = True
    triton_semantic.TritonSemantic.to_tensor = _to_tensor


_patch_constexpr_tensor_to_tensor()


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


_AST_TO_TTIR_OPTION_DEFAULTS = {
    "allowed_dot_input_precisions": ("ieee",),
    "backend_name": "xyz",
    "default_dot_input_precision": "ieee",
    "deprecated_fp8_dot_operand_dtypes": (),
    "extern_libs": None,
    "launch_cooperative_grid": False,
    "max_num_imprecise_acc_default": 0,
    "sanitize_overflow": True,
    "supported_fp8_dtypes": (),
}


def _patch_ast_to_ttir_options():
    current_impl = triton_codegen.ast_to_ttir
    if getattr(current_impl, "_ttx_ast_to_ttir_options_compat", False):
        return

    def _options_with_defaults(options):
        missing = {name: value for name, value in _AST_TO_TTIR_OPTION_DEFAULTS.items() if not hasattr(options, name)}
        if not missing:
            return options
        try:
            for name, value in missing.items():
                setattr(options, name, value)
            return options
        except (AttributeError, TypeError):
            option_values = {}
            try:
                option_values.update(vars(options))
            except TypeError:
                pass
            proxy = types.SimpleNamespace(**option_values)
            for name, value in missing.items():
                setattr(proxy, name, value)
            return proxy

    def _ast_to_ttir(fn, src, context, options, codegen_fns, module_map, *args, **kwargs):
        return current_impl(fn, src, context, _options_with_defaults(options), codegen_fns, module_map, *args, **kwargs)

    _ast_to_ttir._ttx_ast_to_ttir_options_compat = True
    triton_codegen.ast_to_ttir = _ast_to_ttir


_patch_ast_to_ttir_options()


def _patch_cdiv():
    if getattr(getattr(tl, "cdiv", None), "_ttx_cdiv_compat", False):
        return

    @tl_core.builtin
    @tl_core._tensor_member_fn
    def _cdiv(x, div, _semantic=None):
        if isinstance(x, tl_core.constexpr):
            x = x.value
        if isinstance(div, tl_core.constexpr):
            div = div.value

        if isinstance(x, numbers.Number) and isinstance(div, numbers.Number):
            if isinstance(x, bool) or isinstance(div, bool):
                raise ValueError("cdiv does not support boolean type")
            if isinstance(x, int) and isinstance(div, int):
                res = x // div
                rem = x % div
                return res + (1 if rem != 0 else 0)
            return int(np.ceil(x / div))

        x = _semantic.to_tensor(x)
        div = _semantic.to_tensor(div)
        x_scalar_type = x.type.scalar
        div_scalar_type = div.type.scalar
        if x_scalar_type.is_bool() or div_scalar_type.is_bool():
            raise ValueError("cdiv does not support boolean type")
        if x_scalar_type.is_int() and div_scalar_type.is_int():
            return _semantic.floordiv(
                _semantic.add(x, _semantic.sub(div, 1, True), True),
                div,
            )

        div_res = _semantic.truediv(x, div)
        cdiv_res = tl_core.tensor(_semantic.builder.create_ceil(div_res.handle), div_res.type)
        return _semantic.cast(cdiv_res, x_scalar_type)

    _cdiv._ttx_cdiv_compat = True
    tl.cdiv = _cdiv
    triton.language.cdiv = _cdiv  # ty:ignore[attr-defined]
    triton_math.cdiv = _cdiv
    triton_standard.cdiv = _cdiv


_patch_cdiv()


def _patch_reduce_max_propagate_nan():
    current_impl = triton_standard.max
    if getattr(current_impl, "_ttx_reduce_max_propagate_nan_compat", False):
        return

    @triton.jit
    def _elementwise_max_propagate_nan(a, b):
        return tl_core.maximum(a, b, propagate_nan=tl_core.PropagateNan.ALL)

    @tl_core._tensor_member_fn
    @triton.jit
    @tl_core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                                   tie_break_arg="return_indices_tie_break_left")
    def _max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False,
             propagate_nan=False):
        input = tl_core._promote_bfloat16_to_float32(input)
        if return_indices:
            if return_indices_tie_break_left:
                return tl_core._reduce_with_indices(
                    input,
                    axis,
                    triton_standard._argmax_combine_tie_break_left,
                    keep_dims=keep_dims,
                )
            return tl_core._reduce_with_indices(
                input,
                axis,
                triton_standard._argmax_combine_tie_break_fast,
                keep_dims=keep_dims,
            )

        if tl_core.constexpr(input.dtype.primitive_bitwidth) < tl_core.constexpr(32):
            if tl_core.constexpr(input.dtype.is_floating()):
                input = input.to(tl_core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(tl_core.int32)

        if propagate_nan:
            return tl_core.reduce(input, axis, _elementwise_max_propagate_nan, keep_dims=keep_dims)
        return tl_core.reduce(input, axis, triton_standard._elementwise_max, keep_dims=keep_dims)

    _max._ttx_reduce_max_propagate_nan_compat = True
    triton_standard.max = _max
    tl.max = _max
    triton.language.max = _max  # ty:ignore[attr-defined]


_patch_reduce_max_propagate_nan()


def _patch_int_scalar_specialization():
    current_impl = triton_jit.create_function_from_signature
    if getattr(current_impl, "_ttx_int_scalar_constexpr_compat", False):
        return

    def _create_function_from_signature(sig, kparams, backend):
        binder = current_impl(sig, kparams, backend)

        def _dynamic_func(*args, **kwargs):
            params, specialization, options = binder(*args, **kwargs)
            for i, kp in enumerate(kparams):
                if kp.is_constexpr:
                    continue
                value = params[kp.name]
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    continue
                specialization[i] = ("constexpr", value)
            return params, specialization, options

        return _dynamic_func

    _create_function_from_signature._ttx_int_scalar_constexpr_compat = True
    triton_jit.create_function_from_signature = _create_function_from_signature


_patch_int_scalar_specialization()


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


_DEVICE_PRINT_PATTERN = re.compile(r"(?<!static_)device_print\s*\(")
_DEVICE_PRINT_PREFIX_PATTERN = re.compile(r'device_print\(\s*([\'"])(.*?)\1')


def _is_interpreter_builder(builder):
    return builder.__class__.__module__.startswith("triton.runtime.interpreter")


def _device_print_count(jit_fn):
    raw_src = getattr(jit_fn, "raw_src", "")
    if raw_src:
        if isinstance(raw_src, (list, tuple)):
            raw_src = "".join(raw_src)
        return len(_DEVICE_PRINT_PATTERN.findall(raw_src))
    try:
        return len(_DEVICE_PRINT_PATTERN.findall(inspect.getsource(jit_fn.fn)))
    except (OSError, TypeError):
        return 0


def _synthesize_ttadapter_asm(jit_fn):
    count = _device_print_count(jit_fn)
    if count == 0:
        return ""
    return "\n".join(f"call @triton_print_{i}" for i in range(count))


def _device_print_prefixes(jit_fn):
    raw_src = getattr(jit_fn, "raw_src", "")
    if isinstance(raw_src, (list, tuple)):
        raw_src = "".join(raw_src)
    if not raw_src:
        try:
            raw_src = inspect.getsource(jit_fn.fn)
        except (OSError, TypeError):
            return []
    return [match.group(2) for match in _DEVICE_PRINT_PREFIX_PATTERN.finditer(raw_src)]


def _patch_device_print():
    current_impl = triton_semantic.TritonSemantic.device_print
    if getattr(current_impl, "_ttx_device_print_compat", False):
        return

    def _device_print(self, prefix: str, args, hex: bool):
        if _is_interpreter_builder(self.builder):
            return current_impl(self, prefix, args, hex)
        return None

    _device_print._ttx_device_print_compat = True
    triton_semantic.TritonSemantic.device_print = _device_print


_patch_device_print()


def _patch_debug_barrier():
    current_impl = triton_semantic.TritonSemantic.debug_barrier
    if getattr(current_impl, "_ttx_debug_barrier_compat", False):
        return

    def _debug_barrier(self):
        if _is_interpreter_builder(self.builder):
            return current_impl(self)
        return self.tensor(None, tl.void)

    _debug_barrier._ttx_debug_barrier_compat = True
    triton_semantic.TritonSemantic.debug_barrier = _debug_barrier


_patch_debug_barrier()


def _patch_erf():
    current_impl = triton_math.erf
    if getattr(current_impl, "_ttx_erf_compat", False):
        return

    @tl_core.builtin
    @triton_math._check_dtype(dtypes=["fp32", "fp64"])
    @triton_math._add_math_1arg_docstr("error function")
    @tl_core._tensor_member_fn
    def _erf(x, _semantic=None):
        if _is_interpreter_builder(_semantic.builder):
            return current_impl(x, _semantic=_semantic)
        return _approx_erf_semantic(x, _semantic)

    _erf._ttx_erf_compat = True
    triton_math.erf = _erf
    tl.erf = _erf


_patch_erf()


def _patch_jit_run_device_print():
    current_impl = JITFunction.run
    if getattr(current_impl, "_ttx_device_print_runtime_compat", False):
        return

    def _run(self, *args, grid, warmup, **kwargs):
        kernel = current_impl(self, *args, grid=grid, warmup=warmup, **kwargs)
        if warmup or os.environ.get("TRITON_DEVICE_PRINT") != "1":
            return kernel

        device_print_count = _device_print_count(self)
        if device_print_count == 0:
            return kernel

        if kernel is not None and "ttadapter" not in kernel.asm:
            kernel.asm["ttadapter"] = _synthesize_ttadapter_asm(self)

        try:
            interpreter = getattr(self, "_ttx_device_print_interpreter", None)
            if interpreter is None:
                from triton.runtime.interpreter import InterpretedFunction

                interpreter = InterpretedFunction(
                    self.fn,
                    version=self.version,
                    do_not_specialize=self.do_not_specialize,
                    do_not_specialize_on_alignment=self.do_not_specialize_on_alignment,
                    debug=self.debug,
                    noinline=self.noinline,
                    repr=self._repr,
                    launch_metadata=self.launch_metadata,
                )
                self._ttx_device_print_interpreter = interpreter
            interpreter.run(*args, grid=grid, warmup=False, **kwargs)
        except Exception:
            pass

        for prefix in _device_print_prefixes(self):
            print(prefix)

        return kernel

    _run._ttx_device_print_runtime_compat = True
    JITFunction.run = _run


_patch_jit_run_device_print()


def _approx_erf_semantic(x, _semantic):
    x = _semantic.to_tensor(x)
    abs_x = _semantic.tensor(_semantic.builder.create_fabs(x.handle), x.type)
    zero = _semantic.full(abs_x.shape, 0.0, abs_x.type.scalar)
    one = _semantic.full(abs_x.shape, 1.0, abs_x.type.scalar)
    t = _semantic.fdiv(one, _semantic.add(one, _semantic.mul(abs_x, 0.3275911, True), True), False)
    poly = _semantic.sub(_semantic.mul(t, 1.061405429, True), 1.453152027, True)
    poly = _semantic.add(_semantic.mul(poly, t, True), 1.421413741, True)
    poly = _semantic.sub(_semantic.mul(poly, t, True), 0.284496736, True)
    poly = _semantic.add(_semantic.mul(poly, t, True), 0.254829592, True)
    poly = _semantic.mul(poly, t, True)
    neg_abs_sq = _semantic.minus(_semantic.mul(abs_x, abs_x, True))
    exp_neg_abs_sq = _semantic.tensor(_semantic.builder.create_exp(neg_abs_sq.handle), neg_abs_sq.type)
    y = _semantic.sub(one, _semantic.mul(poly, exp_neg_abs_sq, True), True)
    return _semantic.where(_semantic.less_than(x, zero), _semantic.minus(y), y)


@tl_core.builtin
def _buffer_alloc(etype, shape, _address_space=None, is_mem_unique=False, _builder=None, _semantic=None):
    del _address_space, is_mem_unique, _builder
    return tl_core.full(shape, 0, etype, _semantic=_semantic)


_buffer_language_core = types.ModuleType("triton.extension.buffer.language.core")
_buffer_language_core.address_space = xyz_extension._FakeAddressSpace
_buffer_language_core.buffer = tl_core.tensor
_buffer_language_core.alloc = _buffer_alloc
_buffer_language_core.__all__ = ["address_space", "buffer", "alloc"]

_buffer_language = types.ModuleType("triton.extension.buffer.language")
_buffer_language.__path__ = []
_buffer_language.core = _buffer_language_core
_buffer_language.address_space = xyz_extension._FakeAddressSpace
_buffer_language.buffer = tl_core.tensor
_buffer_language.alloc = _buffer_alloc
_buffer_language.__all__ = ["core", "address_space", "buffer", "alloc"]

_buffer_pkg = types.ModuleType("triton.extension.buffer")
_buffer_pkg.__path__ = []
_buffer_pkg.language = _buffer_language

_extension_pkg = types.ModuleType("triton.extension")
_extension_pkg.__path__ = []
_extension_pkg.buffer = _buffer_pkg

setattr(triton, "extension", _extension_pkg)

_fake_ascend_ir = types.ModuleType("triton._C.libtriton.ascend.ir")
_fake_ascend_ir.load_dialects = lambda context: None

_fake_ascend_pkg = types.ModuleType("triton._C.libtriton.ascend")
_fake_ascend_pkg.ir = _fake_ascend_ir

setattr(triton_libtriton, "ascend", _fake_ascend_pkg)


@triton.jit
def _gather_2d_simd(src_ptr, index_ptr, out_ptr, m_size, n_size, k_size, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):  # ty:ignore
    del XBLOCK_SUB
    row_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    col_offsets = tl.arange(0, k_size)[None, :]
    row_mask = row_offsets < m_size
    col_mask = col_offsets < k_size
    mask = row_mask & col_mask
    gathered_idx = tl.load(index_ptr + row_offsets * k_size + col_offsets, mask=mask, other=0)
    gathered = tl.load(src_ptr + row_offsets * n_size + gathered_idx, mask=mask, other=0.0)
    tl.store(out_ptr + row_offsets * k_size + col_offsets, gathered, mask=mask)

setattr(triton.language.extra, "cann", triton.language.extra.xyz)
setattr(triton.language.extra, "ascend", triton.language.extra.xyz)

_xyz_kernels = types.ModuleType("triton.language.extra.kernels")
_xyz_kernels.gather_2d_simd = _gather_2d_simd
_xyz_kernels.__all__ = ["gather_2d_simd"]
setattr(triton.language.extra, "kernels", _xyz_kernels)


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
sys.modules["triton.language.extra.cann.extension"] = xyz_extension  # ty:ignore
sys.modules["triton.language.extra.ascend"] = triton.language.extra.xyz  # ty:ignore
sys.modules["triton.language.extra.ascend.libdevice"] = xyz_libdevice  # ty:ignore
sys.modules["triton.language.extra.ascend.extension"] = xyz_extension  # ty:ignore
sys.modules["triton.language.extra.kernels"] = _xyz_kernels
sys.modules["triton.language.extra.xyz.extension"] = xyz_extension
sys.modules["triton.extension"] = _extension_pkg
sys.modules["triton.extension.buffer"] = _buffer_pkg
sys.modules["triton.extension.buffer.language"] = _buffer_language
sys.modules["triton.extension.buffer.language.core"] = _buffer_language_core
sys.modules["triton._C.libtriton.ascend"] = _fake_ascend_pkg
sys.modules["triton._C.libtriton.ascend.ir"] = _fake_ascend_ir
