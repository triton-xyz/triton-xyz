import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.layernorm import layer_norm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_backward"),
    key=["M", "N"],
)
@triton.jit
def layer_norm_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    has_w: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
    row_mask = pid < M

    dY_ptr = dY + pid * N
    X_ptr = X + pid * N
    dX_ptr = dX + pid * N
    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid

    mean = tl.load(Mean_ptr, mask=row_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr, mask=row_mask, other=0.0).to(tl.float32)

    dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        if has_w:
            w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
        else:
            w = 1.0
        dx_hat = dy * w
        dx_part2 += dx_hat
        dx_part3 += dx_hat * x_hat

    dx_2 = tl.sum(dx_part2, axis=1)[:, None]
    dx_3 = tl.sum(dx_part3, axis=1)[:, None]

    for off in range(0, N, BLOCK_COL_SIZE):
        cols = off + tl.arange(0, BLOCK_COL_SIZE)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        dy = tl.load(dY_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if has_w:
            w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
        else:
            w = 1.0
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = rstd * (dx_hat - (dx_2 + x_hat * dx_3) / N)
        tl.store(dX_ptr + cols[None, :], dx, mask=mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        dy = tl.load(dY + rows * N, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + rows * N, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
        accB += dy
    dw = tl.sum(accW, axis=0)
    db = tl.sum(accB, axis=0)
    tl.store(dW + pid, dw, mask=col_mask)
    tl.store(dB + pid, db, mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def weight_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    X += pid[None, :]
    accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        dy = tl.load(dY + rows * N, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + rows * N, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + rows, mask=row_mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        x_hat = x * rstd
        accW += dy * x_hat
    dw = tl.sum(accW, axis=0)
    tl.store(dW + pid, dw, mask=col_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("weight_bias_backward"),
    key=["N"],
)
@triton.jit
def bias_backward_kernel(
    dY,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = ext.program_id(0) * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    col_mask = pid < N
    dY += pid[None, :]
    accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for off in range(0, M, BLOCK_ROW_SIZE):
        rows = off + tl.arange(0, BLOCK_ROW_SIZE)[:, None]
        row_mask = rows < M
        mask = row_mask & col_mask[None, :]
        dy = tl.load(dY + rows * N, mask=mask, other=0.0).to(tl.float32)
        accB += dy
    db = tl.sum(accB, axis=0)
    tl.store(dB + pid, db, mask=col_mask)


def layer_norm_backward(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight=None,
    bias=None,
    output_mask=None,
):
    logger.debug("GEMS_XYZ LAYERNORM BACKWARD")

    grad_out = grad_out.contiguous()
    input = input.contiguous()
    mean = mean.contiguous()
    rstd = rstd.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    M = input.shape[0]
    N = input.numel() // M

    if output_mask[0]:
        in_grad = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_ROW_SIZE"]), 1, 1)
        has_w = weight is not None
        with torch_device_fn.device(input.device):
            layer_norm_backward_kernel[grid](
                grad_out, input, weight, mean, rstd, in_grad, M, N, has_w
            )
    else:
        in_grad = None

    if output_mask[1] is False and output_mask[2] is False:
        return in_grad, None, None

    weight_grad = torch.empty_like(weight) if output_mask[1] else None
    bias_grad = torch.empty_like(bias) if output_mask[2] else None
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_COL_SIZE"]), 1, 1)
    with torch_device_fn.device(input.device):
        if output_mask[1] and output_mask[2]:
            weight_bias_backward_kernel[grid](
                grad_out, input, mean, rstd, weight_grad, bias_grad, M, N
            )
        elif output_mask[1]:
            weight_backward_kernel[grid](grad_out, input, mean, rstd, weight_grad, M, N)
        else:
            bias_backward_kernel[grid](grad_out, bias_grad, M, N)

    return in_grad, weight_grad, bias_grad
