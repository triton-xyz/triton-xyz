import pytest
import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def basic_addptr_1d_kernel(src_ptr, dst_ptr):
    offsets = tl.arange(0, 4)
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, values)


@triton.jit
def masked_1d_kernel(src_ptr, dst_ptr, limit: int):
    offsets = tl.arange(0, 8)
    mask = offsets < limit
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def block_ptr_basic_kernel(base_ptr):
    ptr = tl.make_block_ptr(
        base=base_ptr,
        shape=(4, 4),
        strides=(4, 1),
        offsets=(0, 0),
        block_shape=(4, 4),
        order=(1, 0),
    )
    advanced_ptr = tl.advance(ptr, (0, 1))
    values = tl.load(advanced_ptr)
    tl.store(ptr, values)


@triton.jit
def gather_scatter_2d_kernel(src_ptr, idx_ptr, dst_ptr):
    linear = tl.arange(0, 16)
    rows = linear // 4
    cols = linear % 4

    gather_rows = tl.load(idx_ptr + rows)
    src_offsets = gather_rows * 4 + cols
    values = tl.load(src_ptr + src_offsets)

    tl.store(dst_ptr + linear, values)


@triton.jit
def scalar_addptr_splat_kernel(src_ptr, dst_ptr, base_offset: int):
    base_ptr = src_ptr + base_offset
    offsets = tl.arange(0, 4)
    values = tl.load(base_ptr + offsets)
    tl.store(dst_ptr + offsets, values)


@triton.jit
def row_major_2d_kernel(src_ptr, dst_ptr):
    rows = tl.arange(0, 2)
    cols = tl.arange(0, 4)
    offsets = rows[:, None] * 4 + cols[None, :]
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, values)


def test_basic_addptr_1d():
    src = torch.arange(4, device=DEVICE, dtype=torch.float32)
    dst = torch.empty_like(src)

    basic_addptr_1d_kernel[(1,)](src, dst)

    torch.testing.assert_close(dst, src)


@pytest.mark.parametrize("limit", [-3, 0, 3, 8, 13])
def test_masked_1d(limit: int):
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)

    masked_1d_kernel[(1,)](src, dst, limit)

    expected = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)
    clipped = max(0, min(limit, 8))
    expected[:clipped] = src[:clipped]
    torch.testing.assert_close(dst, expected)


def test_block_ptr_basic():
    values = torch.arange(17, device=DEVICE, dtype=torch.float16)
    expected = values.clone()
    expected[:16] = values[1:17]

    block_ptr_basic_kernel[(1,)](values)

    torch.testing.assert_close(values, expected)


def test_gather_scatter_2d():
    src = torch.arange(16, device=DEVICE, dtype=torch.float32)
    idx = torch.tensor([2, 0, 3, 1], device=DEVICE, dtype=torch.int32)
    dst = torch.empty((16,), device=DEVICE, dtype=torch.float32)

    gather_scatter_2d_kernel[(1,)](src, idx, dst)

    expected = src.reshape(4, 4)[idx.to(torch.long)].reshape(-1)
    torch.testing.assert_close(dst, expected)


@pytest.mark.parametrize("base_offset", [0, 2, 5])
def test_scalar_addptr_splat(base_offset: int):
    src = torch.arange(12, device=DEVICE, dtype=torch.float32)
    dst = torch.empty((4,), device=DEVICE, dtype=torch.float32)

    scalar_addptr_splat_kernel[(1,)](src, dst, base_offset)

    expected = src[base_offset : base_offset + 4]
    torch.testing.assert_close(dst, expected)


def test_row_major_2d():
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.empty_like(src)

    row_major_2d_kernel[(1,)](src, dst)

    torch.testing.assert_close(dst, src)


if __name__ == "__main__":
    # test_gather_scatter_2d()
    # test_row_major_2d()
    # test_block_ptr_basic()
    test_masked_1d(8)
