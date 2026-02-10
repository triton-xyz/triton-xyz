import torch

import triton
import triton.language as tl


DEVICE: torch.device = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def masked_gather_scatter_kernel(src_ptr, dst_ptr, limit: int):
    offsets = tl.arange(0, 4)
    mask = offsets < limit
    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def offset_width_upgrade_kernel(src_ptr, dst_ptr):
    range_i32 = tl.arange(0, 4)
    offsets_i64 = range_i32.to(tl.int64) + 5
    values = tl.load(src_ptr + offsets_i64)
    tl.store(dst_ptr + range_i32, values)


@triton.jit
def loop_ptr_iter_args_kernel(src_ptr, dst_ptr, n_iters: int):
    range_i32 = tl.arange(0, 4)
    in_ptrs = src_ptr + range_i32
    out_ptrs = dst_ptr + range_i32
    for _ in range(n_iters):
        values = tl.load(in_ptrs)
        tl.store(out_ptrs, values)
        in_ptrs = in_ptrs + range_i32
        out_ptrs = out_ptrs + range_i32


@triton.jit
def make_tensor_ptr_add_base_kernel(src_ptr, dst_ptr):
    pid = tl.program_id(axis=0)
    base_offset = pid * 8
    base_ptr = src_ptr + base_offset

    src_tptr = tl.make_block_ptr(
        base=base_ptr,
        shape=(4,),
        strides=(1,),
        offsets=(1,),
        block_shape=(4,),
        order=(0,),
    )
    values = tl.load(src_tptr)

    dst_tptr = tl.make_block_ptr(
        base=dst_ptr,
        shape=(4,),
        strides=(1,),
        offsets=(0,),
        block_shape=(4,),
        order=(0,),
    )
    tl.store(dst_tptr, values)


def test_masked_gather_scatter():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE, dtype=torch.float32)
    dst = torch.full((4,), -1.0, device=DEVICE, dtype=torch.float32)
    limit = 3

    masked_gather_scatter_kernel[(1,)](src, dst, limit)

    expected = torch.tensor([1.0, 2.0, 3.0, -1.0], device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(dst, expected)


def test_offset_width_upgrade():
    src = torch.arange(0, 16, device=DEVICE, dtype=torch.float32)
    dst = torch.full((4,), -1.0, device=DEVICE, dtype=torch.float32)

    offset_width_upgrade_kernel[(1,)](src, dst)

    torch.testing.assert_close(dst, src[5:9])


def test_loop_ptr_iter_args():
    n_iters = 4
    size = 32
    src = torch.arange(0, size, device=DEVICE, dtype=torch.float32)
    dst = torch.full((size,), -1.0, device=DEVICE, dtype=torch.float32)

    loop_ptr_iter_args_kernel[(1,)](src, dst, n_iters)

    expected = torch.full((size,), -1.0, device=DEVICE, dtype=torch.float32)
    range_i64 = torch.arange(0, 4, device=DEVICE, dtype=torch.int64)
    in_offsets = range_i64.clone()
    out_offsets = range_i64.clone()
    for _ in range(n_iters):
        expected[out_offsets] = src[in_offsets]
        in_offsets = in_offsets + range_i64
        out_offsets = out_offsets + range_i64

    torch.testing.assert_close(dst, expected)


def test_make_tensor_ptr_add_base():
    src = torch.arange(0, 16, device=DEVICE, dtype=torch.float16)
    dst = torch.zeros((4,), device=DEVICE, dtype=torch.float16)

    make_tensor_ptr_add_base_kernel[(1,)](src, dst)

    expected = src[1:5]
    torch.testing.assert_close(dst, expected)


if __name__ == "__main__":
    # test_make_tensor_ptr_add_base()
    test_loop_ptr_iter_args()
