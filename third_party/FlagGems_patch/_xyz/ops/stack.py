import logging
from typing import List, Tuple, Union

import torch

logger = logging.getLogger(__name__)

_COPY_REDISPATCH_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_XYZ STACK")
    if len(tensors) == 0:
        raise RuntimeError("stack expected a non-empty TensorList")

    inp0_shape = list(tensors[0].shape)
    inp0_dim = tensors[0].dim()
    if dim < -inp0_dim - 1 or dim > inp0_dim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -inp0_dim - 1, inp0_dim, dim
            )
        )

    for i, tensor in enumerate(tensors[1:], start=1):
        if list(tensor.shape) != inp0_shape:
            raise RuntimeError(
                f"stack expects each tensor to be equal size, but got {inp0_shape} at entry 0 and {list(tensor.shape)} at entry {i}"
            )

    if dim < 0:
        dim += inp0_dim + 1

    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)

    device = tensors[0].device
    out_shape = inp0_shape[:dim] + [len(tensors)] + inp0_shape[dim:]
    out = torch.empty(out_shape, dtype=dtype, device=device)

    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        torch.ops.aten.copy_.default.redispatch(
            _COPY_REDISPATCH_KEYSET, out.select(dim, i), tensor, False
        )

    return out
