import json
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
def kernel_with_nested_scope(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    with pl.scope("load_pair"):
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x + y, mask=mask)


def _walk(node):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)


def _run_cpu_instrumentation_runtime(output_dir: str):
    profile_path = os.path.join(output_dir, "cpu_instrumentation_runtime")

    size = 1024
    x = torch.ones(size, device=DEVICE, dtype=torch.float32)
    y = torch.ones(size, device=DEVICE, dtype=torch.float32)

    session = proton.start(profile_path, backend="cpu", hook=CPUInstrumentationHook())
    try:
        kernel_with_nested_scope[(1, 1, 1)](x, y, size, BLOCK_SIZE=1024)
    finally:
        proton.finalize(session)

    with open(profile_path + ".hatchet", "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = list(_walk(data[0]))
    kernel_frame = next(
        frame
        for frame in frames
        if frame["frame"]["name"] == "kernel_with_nested_scope" and frame["metrics"].get("time (ns)", 0) > 0
    )
    load_pair_frame = next(
        frame for frame in frames if frame["frame"]["name"] == "load_pair" and frame["metrics"].get("time (ns)", 0) > 0
    )
    assert kernel_frame["metrics"]["time (ns)"] > 0
    assert load_pair_frame["metrics"]["time (ns)"] > 0
    torch.testing.assert_close(x.to("cpu"), torch.full((size,), 2.0))


def test_cpu_instrumentation_runtime():
    _run_cpu_instrumentation_runtime(os.getenv("TRITON_HOME", os.getcwd()))


if __name__ == "__main__":
    _run_cpu_instrumentation_runtime(os.getenv("TRITON_HOME", os.getcwd()))
