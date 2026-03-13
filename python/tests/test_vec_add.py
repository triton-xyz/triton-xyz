try:
    import pytest
except ModuleNotFoundError:

    class _Mark:
        def parametrize(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    class _PytestStub:
        mark = _Mark()

    pytest = _PytestStub()

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, BLOCK_SIZE):
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output


def print_lastn(tensor, prefix, lastn=32):
    last_32 = tensor.flatten()[-32:]
    # last_32 = tensor.flatten()[:32]
    formatted_str = ", ".join([f"{x:.2f}" for x in last_32.tolist()])
    print(f"{prefix}: [{formatted_str}]", flush=True)


@pytest.mark.parametrize("BLOCK_SIZE", [8192])
def test_add(BLOCK_SIZE, log=False):
    device = DEVICE
    size = 8192 * 8 + 8192 * 5
    dtype = torch.float32
    x = torch.ones(size, device=device, dtype=dtype)
    y = torch.ones(size, device=device, dtype=dtype)
    output = torch.zeros(size, device=device, dtype=dtype)
    output_torch = x + y
    if log:
        print_lastn(output_torch, "output_torch")
    output_triton = add(x, y, output, BLOCK_SIZE)
    if log:
        print_lastn(output_triton, "output_triton")
    torch.testing.assert_close(output_triton.to("cpu"), output_torch.to("cpu"))


if __name__ == "__main__":
    test_add(BLOCK_SIZE=8192, log=True)
