import torch

import triton
import triton.language as tl


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def atomic_add_xchg_kernel(ptr, off: int, val: int, mask: tl.int1):
    old = tl.atomic_add(ptr + off, val, mask=mask)
    prev = tl.atomic_xchg(ptr + off, old)
    tl.store(ptr + 8, old + prev + val)


@triton.jit
def atomic_cas_kernel(ptr, off: int, cmp: int, val: int):
    old = tl.atomic_cas(ptr + off, cmp, val)
    tl.store(ptr + 9, old)


@triton.jit
def indirect_reindex_2d_kernel(src_ptr, dst_ptr, row_idx_ptr, col_idx_ptr):
    rows = tl.load(row_idx_ptr + tl.arange(0, 2))
    cols = tl.load(col_idx_ptr + tl.arange(0, 4))

    src_offsets = rows[:, None] * 4 + cols[None, :]
    values = tl.load(src_ptr + src_offsets)

    dst_rows = tl.arange(0, 2)
    dst_cols = tl.arange(0, 4)
    dst_offsets = dst_rows[:, None] * 4 + dst_cols[None, :]
    tl.store(dst_ptr + dst_offsets, values)


@triton.jit
def loop_indirect_seed_kernel(src_ptr, dst_ptr, idx_ptr, n_iters: int):
    idx = tl.load(idx_ptr + tl.arange(0, 4))
    in_ptrs = src_ptr + idx
    out_ptrs = dst_ptr + idx
    for _ in range(n_iters):
        values = tl.load(in_ptrs)
        tl.store(out_ptrs, values)
        in_ptrs = in_ptrs + 4
        out_ptrs = out_ptrs + 4


@triton.jit
def loop_indirect_recurrence_kernel(src_ptr, dst_ptr, idx_ptr, n_iters: int):
    idx = tl.load(idx_ptr + tl.arange(0, 4))
    in_ptrs = src_ptr + idx
    out_ptrs = dst_ptr + idx
    for _ in range(n_iters):
        values = tl.load(in_ptrs)
        tl.store(out_ptrs, values)
        in_ptrs = in_ptrs + 4
        out_ptrs = out_ptrs + 4
        in_ptrs = in_ptrs + idx
        out_ptrs = out_ptrs + idx


@triton.jit
def wrap_dynamic_mask_kernel(src_ptr, dst_ptr, boundary: int):
    base = tl.arange(0, 4)
    src_offsets = (base + 1) % boundary
    dst_offsets = (base + 2) % boundary
    mask = base != 1
    values = tl.load(src_ptr + src_offsets, mask=mask, other=-3.0)
    tl.store(dst_ptr + dst_offsets, values, mask=mask)


def test_atomic_add_xchg_masked_true():
    values = torch.arange(16, device=DEVICE, dtype=torch.int32)

    atomic_add_xchg_kernel[(1,)](values, 3, 7, True)

    expected = torch.arange(16, device=DEVICE, dtype=torch.int32)
    old = int(expected[3].item())
    expected[8] = old + (old + 7) + 7
    torch.testing.assert_close(values, expected)


def test_atomic_add_xchg_masked_false():
    values = torch.arange(16, device=DEVICE, dtype=torch.int32)

    atomic_add_xchg_kernel[(1,)](values, 3, 7, False)

    expected = torch.arange(16, device=DEVICE, dtype=torch.int32)
    old = int(expected[3].item())
    expected[8] = old + old + 7
    torch.testing.assert_close(values, expected)


def test_atomic_cas_scalar():
    hit = torch.arange(16, device=DEVICE, dtype=torch.int32)
    miss = torch.arange(16, device=DEVICE, dtype=torch.int32)

    atomic_cas_kernel[(1,)](hit, 5, 5, 99)
    atomic_cas_kernel[(1,)](miss, 5, 77, 99)

    expected_hit = torch.arange(16, device=DEVICE, dtype=torch.int32)
    expected_hit[9] = expected_hit[5]
    expected_hit[5] = 99

    expected_miss = torch.arange(16, device=DEVICE, dtype=torch.int32)
    expected_miss[9] = expected_miss[5]

    torch.testing.assert_close(hit, expected_hit)
    torch.testing.assert_close(miss, expected_miss)


def test_indirect_reindex_2d():
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)
    row_idx = torch.tensor([1, 0], device=DEVICE, dtype=torch.int32)
    col_idx = torch.tensor([3, 1, 0, 2], device=DEVICE, dtype=torch.int32)

    indirect_reindex_2d_kernel[(1,)](src, dst, row_idx, col_idx)

    expected = src.reshape(2, 4)[row_idx.to(torch.long)][:, col_idx.to(torch.long)]
    torch.testing.assert_close(dst.reshape(2, 4), expected)


def test_loop_indirect_seed():
    src = torch.arange(16, device=DEVICE, dtype=torch.float32)
    dst = torch.full((16,), -1.0, device=DEVICE, dtype=torch.float32)
    idx = torch.tensor([3, 1, 0, 2], device=DEVICE, dtype=torch.int32)

    loop_indirect_seed_kernel[(1,)](src, dst, idx, 2)

    expected = torch.full((16,), -1.0, device=DEVICE, dtype=torch.float32)
    in_offsets = idx.to(torch.long).clone()
    out_offsets = idx.to(torch.long).clone()
    for _ in range(2):
        expected[out_offsets] = src[in_offsets]
        in_offsets = in_offsets + 4
        out_offsets = out_offsets + 4

    torch.testing.assert_close(dst, expected)


def test_loop_indirect_recurrence():
    src = torch.arange(32, device=DEVICE, dtype=torch.float32)
    dst = torch.full((32,), -1.0, device=DEVICE, dtype=torch.float32)
    idx = torch.tensor([0, 1, 2, 3], device=DEVICE, dtype=torch.int32)

    loop_indirect_recurrence_kernel[(1,)](src, dst, idx, 2)

    expected = torch.full((32,), -1.0, device=DEVICE, dtype=torch.float32)
    in_offsets = idx.to(torch.long).clone()
    out_offsets = idx.to(torch.long).clone()
    step = idx.to(torch.long)
    for _ in range(2):
        expected[out_offsets] = src[in_offsets]
        in_offsets = in_offsets + 4
        out_offsets = out_offsets + 4
        in_offsets = in_offsets + step
        out_offsets = out_offsets + step

    torch.testing.assert_close(dst, expected)


def test_wrap_dynamic_mask():
    src = torch.arange(8, device=DEVICE, dtype=torch.float32)
    dst = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)
    boundary = 5

    wrap_dynamic_mask_kernel[(1,)](src, dst, boundary)

    expected = torch.full((8,), -1.0, device=DEVICE, dtype=torch.float32)
    base = torch.arange(4, device=DEVICE, dtype=torch.int64)
    src_offsets = (base + 1) % boundary
    dst_offsets = (base + 2) % boundary
    mask = base != 1
    expected[dst_offsets[mask]] = src[src_offsets[mask]]

    torch.testing.assert_close(dst, expected)
