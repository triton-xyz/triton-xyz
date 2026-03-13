import os

import torch

import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.backends.xyz.proton import CPUInstrumentationHook

DEVICE = triton.runtime.driver.active.get_active_torch_device()
pl.enable_semantic("triton")


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
    with pl.scope("add"):
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, BLOCK_SIZE):
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output


def run_instrumentation():
    output_dir = os.getenv("TRITON_HOME", os.getcwd())
    profile_path = os.path.join(output_dir, "add")

    size = 8192 * 8 + 8192 * 5
    x = torch.ones(size, device=DEVICE, dtype=torch.float32)
    y = torch.ones(size, device=DEVICE, dtype=torch.float32)
    output = torch.zeros(size, device=DEVICE, dtype=torch.float32)

    session = proton.start(profile_path, backend="cpu", hook=CPUInstrumentationHook())
    output_triton = add(x, y, output, 8192)
    proton.finalize(session)

    torch.testing.assert_close(output_triton.to("cpu"), (x + y).to("cpu"))


if __name__ == "__main__":
    run_instrumentation()
