import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl


LIBPROTON = Path(__file__).resolve().parents[2] / "build" / "libproton.so"

if LIBPROTON.exists():
    import triton.profiler as proton
    import triton.profiler.language as pl
    from triton.backends.xyz.proton import CPUInstrumentationHook

    pl.enable_semantic("triton")


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def threaded_profile_kernel(output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.where(mask, offsets.to(tl.float32), 0.0)
    for _ in tl.static_range(0, 64):
        with pl.scope("work"):  # type: ignore[name-defined]
            values = values * 1.0001 + 1.0
    tl.store(output_ptr + offsets, values, mask=mask)


@pytest.mark.skipif(not LIBPROTON.exists(), reason="CPU proton profiling requires build/libproton.so")
def test_cpu_trace_uses_multiple_thread_lanes(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_XYZ_NUM_THREADS", "4")

    num_programs = 256
    block = 256
    n_elements = num_programs * block
    output = torch.empty(n_elements, device=DEVICE, dtype=torch.float32)

    profile_path = tmp_path / "cpu_profile"
    session = proton.start(
        str(profile_path),
        data="trace",
        backend="cpu",
        hook=CPUInstrumentationHook(),
    )
    threaded_profile_kernel[(num_programs,)](output, n_elements, BLOCK=block)
    proton.finalize(session)

    trace_path = profile_path.with_suffix(".chrome_trace")
    trace_objects = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    names = {event["name"] for trace_object in trace_objects for event in trace_object["traceEvents"]}
    tids = {event["tid"] for trace_object in trace_objects for event in trace_object["traceEvents"]}
    assert len(tids) > 1
    assert "threaded_profile_kernel" in names
    assert "work" in names
