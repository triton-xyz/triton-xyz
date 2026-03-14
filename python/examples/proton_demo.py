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
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    with pl.scope("load_row"):
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    with pl.scope("subtract_max"):
        row_minus_max = row - tl.max(row, axis=0)
    with pl.scope("exp"):
        numerator = tl.exp(row_minus_max)
    with pl.scope("sum"):
        denominator = tl.sum(numerator, axis=0)
    with pl.scope("normalize"):
        softmax_output = numerator / denominator
    with pl.scope("store_row"):
        output_row_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)
    with proton.scope("launch.softmax_kernel"):
        softmax_kernel[(n_rows,)](
            output,
            x,
            x.stride(0),
            output.stride(0),
            n_cols,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
    return output


def make_demo_input() -> torch.Tensor:
    torch.manual_seed(0)
    rows, cols = 37, 781
    return torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)


def run_softmax_demo(*, data: str, profile_path: str) -> torch.Tensor:
    x = make_demo_input()
    session = proton.start(
        profile_path,
        data=data,
        backend="cpu",
        hook=CPUInstrumentationHook(),
    )
    with proton.scope("softmax_demo"):
        output_triton = softmax(x)
    proton.finalize(session)

    output_torch = torch.softmax(x, dim=1)
    torch.testing.assert_close(output_triton.to("cpu"), output_torch.to("cpu"))
    return output_triton


def run_chrome_trace_demo():
    output_dir = os.getenv("TRITON_HOME", os.getcwd())
    profile_path = os.path.join(output_dir, "softmax")
    run_softmax_demo(data="trace", profile_path=profile_path)
    print(f"chrome trace written to {profile_path}.chrome_trace")


def run_hatchet_demo():
    output_dir = os.getenv("TRITON_HOME", os.getcwd())
    profile_path = os.path.join(output_dir, "softmax")
    run_softmax_demo(data="tree", profile_path=profile_path)
    print(f"hatchet profile written to {profile_path}.hatchet")


if __name__ == "__main__":
    run_chrome_trace_demo()
    run_hatchet_demo()
