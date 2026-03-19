import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import triton
import triton.language as tl
import triton.profiler as proton
import triton.profiler.language as pl
from triton.backends.xyz.proton import CPUInstrumentationHook

DEVICE = triton.runtime.driver.active.get_active_torch_device()
pl.enable_semantic("triton")
THREAD_NODE_PREFIX = "thread "


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
    rows = max(1024, get_requested_num_threads() * 256)
    cols = 781
    return torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)


def get_requested_num_threads() -> int:
    raw_value = os.getenv("TRITON_XYZ_NUM_THREADS", "1")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 1


def get_expected_min_thread_count() -> int:
    return 2 if get_requested_num_threads() > 1 else 1


@dataclass(frozen=True)
class ProfileSummary:
    path: Path
    thread_count: int
    thread_labels: tuple[str, ...]


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    trace_events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        trace_events.extend(json.loads(line).get("traceEvents", []))
    return trace_events


def _walk_hatchet_tree(node: dict[str, Any], frame_names: set[str], thread_labels: set[str]) -> None:
    frame = node.get("frame", {})
    name = frame.get("name")
    if isinstance(name, str):
        frame_names.add(name)
        if name.startswith(THREAD_NODE_PREFIX):
            thread_labels.add(name)
    for child in node.get("children", []):
        _walk_hatchet_tree(child, frame_names, thread_labels)


def validate_chrome_trace_profile(profile_path: Path) -> ProfileSummary:
    trace_path = profile_path.with_suffix(".chrome_trace")
    trace_events = _load_trace_events(trace_path)
    thread_labels = {str(event["tid"]) for event in trace_events if "tid" in event and event.get("ph") == "X"}
    names = {str(event["name"]) for event in trace_events if "name" in event and event.get("ph") == "X"}
    expected_min_threads = get_expected_min_thread_count()
    assert len(thread_labels) >= expected_min_threads, (
        f"expected at least {expected_min_threads} chrome trace threads, " f"got {len(thread_labels)} from {trace_path}"
    )
    assert "softmax_kernel" in names
    assert "load_row" in names
    return ProfileSummary(
        path=trace_path,
        thread_count=len(thread_labels),
        thread_labels=tuple(sorted(thread_labels)),
    )


def validate_hatchet_profile(profile_path: Path) -> ProfileSummary:
    hatchet_path = profile_path.with_suffix(".hatchet")
    hatchet_tree, _device_metadata = json.loads(hatchet_path.read_text())
    frame_names: set[str] = set()
    thread_labels: set[str] = set()
    _walk_hatchet_tree(hatchet_tree, frame_names, thread_labels)
    expected_min_threads = get_expected_min_thread_count()
    assert len(thread_labels) >= expected_min_threads, (
        f"expected at least {expected_min_threads} hatchet thread nodes, "
        f"got {len(thread_labels)} from {hatchet_path}"
    )
    assert "softmax_kernel" in frame_names
    assert "load_row" in frame_names
    return ProfileSummary(
        path=hatchet_path,
        thread_count=len(thread_labels),
        thread_labels=tuple(sorted(thread_labels)),
    )


def run_softmax_demo(*, data: str, profile_path: Path) -> torch.Tensor:
    x = make_demo_input()
    session = proton.start(
        str(profile_path),
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
    output_dir = Path(os.getenv("TRITON_HOME", os.getcwd()))
    profile_path = output_dir / "softmax"
    run_softmax_demo(data="trace", profile_path=profile_path)
    summary = validate_chrome_trace_profile(profile_path)
    print(f"chrome trace written to {summary.path}, " f"threads={summary.thread_count}")


def run_hatchet_demo():
    output_dir = Path(os.getenv("TRITON_HOME", os.getcwd()))
    profile_path = output_dir / "softmax"
    run_softmax_demo(data="tree", profile_path=profile_path)
    summary = validate_hatchet_profile(profile_path)
    print(f"hatchet profile written to {summary.path}, " f"threads={summary.thread_count}")


if __name__ == "__main__":
    run_chrome_trace_demo()
    run_hatchet_demo()
