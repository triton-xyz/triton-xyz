import subprocess
import pytest
import torch
import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def regular_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def ptr_select_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    even = (offsets % 2) == 0
    ptr0 = src_ptr + offsets
    ptr1 = src_ptr + offsets + 1
    ptrs = tl.where(even, ptr0, ptr1)
    vals = tl.load(ptrs, mask=mask, other=0)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@pytest.mark.skip("skip")
def test_regular_copy():
    n_elements = 256
    src = torch.arange(0, n_elements, dtype=torch.int32, device=DEVICE)
    dst = torch.empty(n_elements, dtype=torch.int32, device=DEVICE)

    grid = (triton.cdiv(n_elements, 64),)
    regular_copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=64)

    torch.testing.assert_close(dst, src)


# @pytest.mark.skip("skip")
def test_pointer_select():
    n_elements = 256
    src = torch.arange(0, n_elements + 1, dtype=torch.int32, device=DEVICE)
    dst = torch.empty(n_elements, dtype=torch.int32, device=DEVICE)

    grid = (triton.cdiv(n_elements, 64),)

    ptr_select_kernel[grid](src, dst, n_elements, BLOCK_SIZE=64)

    # with pytest.raises(subprocess.CalledProcessError) as exc_info:
    #     ptr_select_kernel[grid](src, dst, n_elements, BLOCK_SIZE=64)
    # assert exc_info.value.returncode != 0
    # assert "mlir-opt" in " ".join(exc_info.value.cmd)
