import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir  # ty:ignore
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.language.extra import libdevice

from triton.backends.xyz.compiler import XYZBackend


@triton.jit
def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    y = libdevice.nop(x)  # ty:ignore
    tl.store(out_ptr + offs, y)


def test_make_ir() -> None:
    x = torch.randn(4)
    out = torch.empty_like(x)

    target = GPUTarget("cpu", "cpu", 1)
    backend = XYZBackend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=kernel,
        signature={
            "in_ptr": triton.runtime.jit.mangle_type(x),
            "out_ptr": triton.runtime.jit.mangle_type(out),
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 4},
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    text = str(module)
    print(text)
    assert "tt.nop" in text, text


if __name__ == "__main__":
    test_make_ir()
