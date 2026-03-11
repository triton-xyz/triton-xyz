import types

import triton
import triton.language as tl
import triton.language.core as tl_core


class _FakeAddressSpace:

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def _fake_enum_group(*names):
    return types.SimpleNamespace(**{name: _FakeAddressSpace(name) for name in names})


ascend_address_space = _fake_enum_group("GM", "L1", "L0A", "L0B", "L0C", "UB")
FixpipeDMAMode = _fake_enum_group("NZ2DN", "NZ2ND", "NZ2NZ")
FixpipeDualDstMode = _fake_enum_group("NO_DUAL", "COLUMN_SPLIT", "ROW_SPLIT")
FixpipePreQuantMode = _fake_enum_group("NO_QUANT", "F322BF16", "F322F16", "S322I8")
FixpipePreReluMode = _fake_enum_group("LEAKY_RELU", "NO_RELU", "NORMAL_RELU", "P_RELU")


@tl_core.builtin
def fixpipe(src, dst, dma_mode=None, dual_dst_mode=None, _builder=None, _semantic=None):
    del src, dst, dma_mode, dual_dst_mode, _builder, _semantic
    return None


@triton.jit
def _sum_combine(a, b):  # ty:ignore
    return a + b


@triton.jit
def _or_combine(a, b):  # ty:ignore
    return a | b


def _constexpr_value(value):
    return value.value if isinstance(value, tl_core.constexpr) else value


def _validate_slice_args(full_tensor, offsets, sizes, strides, sub_tensor=None):
    if len(full_tensor.shape) != 1:
        raise NotImplementedError("fake cann.extension only supports rank-1 tensors")
    if len(offsets) != 1 or len(sizes) != 1 or len(strides) != 1:
        raise NotImplementedError("fake cann.extension only supports rank-1 slice metadata")
    slice_size = _constexpr_value(sizes[0])
    stride = _constexpr_value(strides[0])
    if not isinstance(slice_size, int) or slice_size < 1:
        raise ValueError("slice size must be a positive integer")
    if stride != 1:
        raise NotImplementedError("fake cann.extension only supports unit strides")
    if sub_tensor is not None and tuple(sub_tensor.shape) != (slice_size,):
        raise ValueError("slice payload shape does not match requested size")
    return slice_size


@tl_core.builtin
def extract_slice(full_tensor, offsets, sizes, strides, _builder=None, _generator=None, _semantic=None):
    slice_size = _validate_slice_args(full_tensor, offsets, sizes, strides)
    full_size = _constexpr_value(full_tensor.shape[0])
    offset = _semantic.to_tensor(offsets[0])
    local_offset = _semantic.mod(offset, full_size)
    full_idx = tl_core.arange(0, full_size, _semantic=_semantic)
    slice_idx = _semantic.add(local_offset, tl_core.arange(0, slice_size, _semantic=_semantic), True)
    full_vals = tl_core.broadcast_to(tl_core.expand_dims(full_tensor, 0, _semantic=_semantic),
                                     (slice_size, full_size), _semantic=_semantic)
    full_idx = tl_core.broadcast_to(tl_core.expand_dims(full_idx, 0, _semantic=_semantic),
                                    (slice_size, full_size), _semantic=_semantic)
    slice_idx = tl_core.broadcast_to(tl_core.expand_dims(slice_idx, 1, _semantic=_semantic),
                                     (slice_size, full_size), _semantic=_semantic)
    zeros = tl_core.full((slice_size, full_size), 0, full_tensor.type.scalar, _semantic=_semantic)
    selected = _semantic.where(_semantic.equal(full_idx, slice_idx), full_vals, zeros)
    return tl_core.reduce(selected, 1, _sum_combine, _semantic=_semantic, _generator=_generator)


@tl_core.builtin
def insert_slice(full_tensor, sub_tensor, offsets, sizes, strides, _builder=None, _generator=None, _semantic=None):
    slice_size = _validate_slice_args(full_tensor, offsets, sizes, strides, sub_tensor=sub_tensor)
    full_size = _constexpr_value(full_tensor.shape[0])
    offset = _semantic.to_tensor(offsets[0])
    local_offset = _semantic.mod(offset, full_size)
    full_idx = tl_core.arange(0, full_size, _semantic=_semantic)
    slice_idx = _semantic.add(local_offset, tl_core.arange(0, slice_size, _semantic=_semantic), True)
    sub_vals = tl_core.broadcast_to(tl_core.expand_dims(sub_tensor, 0, _semantic=_semantic),
                                    (full_size, slice_size), _semantic=_semantic)
    full_idx = tl_core.broadcast_to(tl_core.expand_dims(full_idx, 1, _semantic=_semantic),
                                    (full_size, slice_size), _semantic=_semantic)
    slice_idx = tl_core.broadcast_to(tl_core.expand_dims(slice_idx, 0, _semantic=_semantic),
                                     (full_size, slice_size), _semantic=_semantic)
    selector = _semantic.equal(full_idx, slice_idx)
    data_zeros = tl_core.full((full_size, slice_size), 0, sub_tensor.type.scalar, _semantic=_semantic)
    inserted = tl_core.reduce(_semantic.where(selector, sub_vals, data_zeros), 1, _sum_combine,
                              _semantic=_semantic, _generator=_generator)
    matched = tl_core.reduce(selector, 1, _or_combine, _semantic=_semantic, _generator=_generator)
    return _semantic.where(matched, inserted, full_tensor)


__all__ = [
    "ascend_address_space",
    "FixpipeDMAMode",
    "FixpipeDualDstMode",
    "FixpipePreQuantMode",
    "FixpipePreReluMode",
    "extract_slice",
    "fixpipe",
    "insert_slice",
]
