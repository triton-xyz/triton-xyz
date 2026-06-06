import logging
import math
import importlib

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_base = importlib.import_module("flag_gems.ops.kron")


@triton.jit
def calculate_batch_indices_kernel(
    batch_indices_ptr,
    batch_size: tl.int64,
    a_batch0: tl.int64,
    a_batch1: tl.int64,
    b_batch0: tl.int64,
    b_batch1: tl.int64,
    out_batch0: tl.int64,
    out_batch1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offset = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offset < batch_size

    out_indice1 = offset % out_batch1
    remaining = offset // out_batch1
    out_indice0 = remaining % out_batch0
    a_idx = out_indice0 // b_batch0
    a_idx = a_idx * a_batch1 + (out_indice1 // b_batch1)
    b_idx = out_indice0 % b_batch0
    b_idx = b_idx * b_batch1 + (out_indice1 % b_batch1)

    a_store_offset = 2 * offset
    b_store_offset = 2 * offset + 1
    tl.store(batch_indices_ptr + a_store_offset, a_idx, mask=mask)
    tl.store(batch_indices_ptr + b_store_offset, b_idx, mask=mask)


def kron(A, B):
    logger.debug("GEMS_XYZ KRON")
    if A.dim() == 0 and B.dim() == 0:
        return A * B

    if A.numel() == 0 or B.numel() == 0:
        A_prepared, B_prepared, out_shape = _base.prepare_tensor_for_kron(A, B)
        output_dtype = torch.promote_types(A.dtype, B.dtype)
        return torch.empty(out_shape, device=A.device, dtype=output_dtype)

    if A.dim() == 0:
        return A.unsqueeze(0) * B
    if B.dim() == 0:
        return A * B.unsqueeze(0)

    A_prepared, B_prepared, out_shape = _base.prepare_tensor_for_kron(A, B)
    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2
    batch_size = math.prod(out_shape[:-2]) if out_shape[:-2] else 1

    output_dtype = torch.promote_types(A.dtype, B.dtype)
    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    C_reshaped = C.view(-1, M, N)
    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()
    a_batch_stride = M1 * N1
    b_batch_stride = M2 * N2
    c_batch_stride = M * N
    if A_prepared.dim() == 4 and B_prepared.dim() == 4:
        batch_indices = torch.empty(batch_size * 2, device=A.device, dtype=torch.int64)
        a_batch0, a_batch1 = A_prepared.shape[:-2]
        b_batch0, b_batch1 = B_prepared.shape[:-2]
        out_batch0 = a_batch0 * b_batch0
        out_batch1 = a_batch1 * b_batch1
        indice_tile_size = 256
        grid_for_indice = (triton.cdiv(batch_size, indice_tile_size),)
        with torch_device_fn.device(A.device):
            calculate_batch_indices_kernel[grid_for_indice](
                batch_indices,
                batch_size,
                a_batch0,
                a_batch1,
                b_batch0,
                b_batch1,
                out_batch0,
                out_batch1,
                BLOCK_SIZE=indice_tile_size,
            )
            grid = lambda meta: (
                batch_size
                * triton.cdiv(M, meta["BLOCK_M"])
                * triton.cdiv(N, meta["BLOCK_N"]),
            )

            _base.kron_kernel[grid](
                A_view,
                B_view,
                C_reshaped,
                batch_indices,
                batch_size,
                M,
                N,
                M1,
                M2,
                N1,
                N2,
                A_view.stride(1),
                A_view.stride(2),
                B_view.stride(1),
                B_view.stride(2),
                C_reshaped.stride(1),
                C_reshaped.stride(2),
                a_batch_stride,
                b_batch_stride,
                c_batch_stride,
            )

    else:
        if batch_size != 1:
            batch_indices = torch.empty(
                batch_size * 2, device=A.device, dtype=torch.int64
            )
            for i in range(batch_size):
                a_idx, b_idx = _base.calculate_indices(
                    i, A_prepared.shape, B_prepared.shape
                )
                batch_indices[i * 2] = a_idx
                batch_indices[i * 2 + 1] = b_idx
            with torch_device_fn.device(A.device):
                grid = lambda meta: (
                    batch_size
                    * triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(N, meta["BLOCK_N"]),
                )
                _base.kron_kernel[grid](
                    A_view,
                    B_view,
                    C_reshaped,
                    batch_indices,
                    batch_size,
                    M,
                    N,
                    M1,
                    M2,
                    N1,
                    N2,
                    A_view.stride(1),
                    A_view.stride(2),
                    B_view.stride(1),
                    B_view.stride(2),
                    C_reshaped.stride(1),
                    C_reshaped.stride(2),
                    a_batch_stride,
                    b_batch_stride,
                    c_batch_stride,
                )
        else:
            with torch_device_fn.device(A.device):
                grid = lambda meta: (
                    batch_size
                    * triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(N, meta["BLOCK_N"]),
                )
                _base.kron_kernel_for_batch_size_1[grid](
                    A_view,
                    B_view,
                    C_reshaped,
                    batch_size,
                    M,
                    N,
                    M1,
                    M2,
                    N1,
                    N2,
                    A_view.stride(1),
                    A_view.stride(2),
                    B_view.stride(1),
                    B_view.stride(2),
                    C_reshaped.stride(1),
                    C_reshaped.stride(2),
                )
    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)

    return C
