import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def wrap_static_1d_kernel(src_ptr, dst_ptr):
    offsets = (tl.arange(0, 8) + 6) % 8
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + tl.arange(0, 8), values)


@triton.jit
def wrap_dynamic_1d_kernel(src_ptr, dst_ptr, boundary: int):
    offsets = (tl.arange(0, 8) + 6) % boundary
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + tl.arange(0, 8), values)


@triton.jit
def wrap_row_2d_kernel(src_ptr, dst_ptr):
    out_rows = tl.arange(0, 2)[:, None]
    src_rows = (out_rows + 1) % 2
    cols = tl.arange(0, 4)[None, :]

    values = tl.load(src_ptr + src_rows * 4 + cols)
    tl.store(dst_ptr + out_rows * 4 + cols, values)


def test_wrap_static_1d():
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)

    wrap_static_1d_kernel[(1,)](src, dst)

    expected = src[torch.tensor([6, 7, 0, 1, 2, 3, 4, 5], device=DEVICE)]
    torch.testing.assert_close(dst, expected)


def test_wrap_dynamic_1d():
    boundary = 8
    src = torch.arange(boundary, device=DEVICE, dtype=torch.float32)
    dst = torch.full((boundary,), -1.0, device=DEVICE, dtype=torch.float32)

    wrap_dynamic_1d_kernel[(1,)](src, dst, boundary)

    offsets = (torch.arange(boundary, device=DEVICE) + 6) % boundary
    torch.testing.assert_close(dst, src[offsets])


def test_wrap_row_2d():
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)

    wrap_row_2d_kernel[(1,)](src, dst)

    expected = src.reshape(2, 4)[torch.tensor([1, 0], device=DEVICE)].reshape(8)
    torch.testing.assert_close(dst, expected)
