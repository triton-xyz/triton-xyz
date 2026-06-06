from __future__ import annotations

from typing import List, Tuple, Union

import torch

TensorList = Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]

_COPY_REDISPATCH_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _copy_into(dst: torch.Tensor, src: torch.Tensor):
    torch.ops.aten.copy_.default.redispatch(_COPY_REDISPATCH_KEYSET, dst, src, False)


def _normalize_cat_inputs(tensors: TensorList, dim: int):
    if len(tensors) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    tensors = list(tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype

    for i in range(len(tensors) - 1, -1, -1):
        if tensors[i].shape == torch.Size([0]):
            tensors.pop(i)

    if len(tensors) == 0:
        return "empty", torch.tensor([], device=device, dtype=dtype)
    if len(tensors) == 1:
        return "single", tensors[0]

    rank = tensors[0].ndim
    if dim < -rank or dim >= rank:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {dim})")
    dim %= rank

    inp0_shape = list(tensors[0].shape)
    for tensor_idx, tensor in enumerate(tensors):
        if tensor.device != device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found at least two devices, {device} and {tensor.device}"
            )
        inp_shape = list(tensor.shape)
        if len(inp_shape) != len(inp0_shape):
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got {len(inp0_shape)} and {len(inp_shape)}"
            )
        for idx, (common_length, length) in enumerate(zip(inp0_shape, inp_shape)):
            if idx != dim and length != common_length:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {common_length} but got size {length} for tensor number "
                    f"{tensor_idx} in the list"
                )

    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    tensors = [tensor.to(dtype) if tensor.dtype != dtype else tensor for tensor in tensors]

    out_shape = list(tensors[0].shape)
    out_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return "multi", (tensors, dim, out_shape, dtype, device)


def cat(tensors: TensorList, dim: int = 0) -> torch.Tensor:
    mode, payload = _normalize_cat_inputs(tensors, dim)
    if mode in ("single", "empty"):
        return payload

    tensors, dim, out_shape, dtype, device = payload
    out = torch.empty(out_shape, dtype=dtype, device=device)
    offset = 0
    for tensor in tensors:
        size = tensor.shape[dim]
        if size:
            _copy_into(out.narrow(dim, offset, size), tensor)
        offset += size
    return out


def cat_out(tensors: TensorList, dim: int = 0, *, out: torch.Tensor) -> torch.Tensor:
    result = cat(tensors, dim)
    out.resize_(result.shape)
    if out.dtype != result.dtype:
        result = result.to(out.dtype)
    _copy_into(out, result)
    return out


def hstack(tensors: TensorList) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("hstack expected a non-empty TensorList")

    normalized = []
    for tensor in tensors:
        normalized.append(tensor.reshape(1) if tensor.ndim == 0 else tensor)
    dim = 0 if normalized[0].ndim == 1 else 1
    return cat(normalized, dim)


def _atleast_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def vstack(tensors: TensorList) -> torch.Tensor:
    if len(tensors) == 0:
        raise RuntimeError("vstack expected a non-empty TensorList")
    return cat([_atleast_2d(tensor) for tensor in tensors], 0)
