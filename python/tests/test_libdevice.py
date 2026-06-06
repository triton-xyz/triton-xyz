import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir  # ty:ignore
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.language.extra import libdevice

from triton.backends.xyz.compiler import XYZBackend


@triton.jit
def nop_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    y = libdevice.nop(x)  # ty:ignore
    tl.store(out_ptr + offs, y)


@triton.jit
def pow_tensor_scalar_kernel(
    in_ptr, out_ptr, exponent: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    y = libdevice.pow(x, exponent)  # ty:ignore
    tl.store(out_ptr + offs, y)


@triton.jit
def pow_scalar_tensor_kernel(in_ptr, out_ptr, base: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    exponent = tl.load(in_ptr + offs)
    y = libdevice.pow(base, exponent)  # ty:ignore
    tl.store(out_ptr + offs, y)


@triton.jit
def fmod_tensor_scalar_kernel(
    in_ptr, out_ptr, divisor: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    y = libdevice.fmod(x, divisor)  # ty:ignore
    tl.store(out_ptr + offs, y)


@triton.jit
def fmod_tensor_tensor_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    divisor = x + 3.0
    y = libdevice.fmod(x, divisor)  # ty:ignore
    tl.store(out_ptr + offs, y)


@triton.jit
def trunc_div_rz_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + offs)
    divisor = x + 3.0
    y = libdevice.trunc(libdevice.div_rz(x, divisor))  # ty:ignore
    tl.store(out_ptr + offs, y)


def make_ir(fn, constexprs):
    x = torch.randn(4)
    out = torch.empty_like(x)

    target = GPUTarget("cpu", "cpu", 1)
    backend = XYZBackend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(
        fn=fn,
        signature={
            "in_ptr": triton.runtime.jit.mangle_type(x),
            "out_ptr": triton.runtime.jit.mangle_type(out),
            **{name: "constexpr" for name in constexprs if name != "BLOCK"},
            "BLOCK": "constexpr",
        },
        constexprs=constexprs,
    )
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return str(module)


def test_make_ir() -> None:
    text = make_ir(nop_kernel, {"BLOCK": 4})
    print(text)
    assert "tt.nop" in text, text


def test_pow_tensor_scalar_splats_scalar() -> None:
    text = make_ir(pow_tensor_scalar_kernel, {"exponent": 0.5, "BLOCK": 4})
    print(text)
    assert "arith.constant dense<5.000000e-01> : tensor<4xf32>" in text, text
    assert "tt.extern_elementwise" in text, text
    assert 'symbol = "__nv_powf"' in text, text
    assert "tensor<4xf32>, f32" not in text, text
    assert "f32, tensor<4xf32>" not in text, text


def test_pow_scalar_tensor_splats_scalar() -> None:
    text = make_ir(pow_scalar_tensor_kernel, {"base": 0.5, "BLOCK": 4})
    print(text)
    assert "arith.constant dense<5.000000e-01> : tensor<4xf32>" in text, text
    assert "tt.extern_elementwise" in text, text
    assert 'symbol = "__nv_powf"' in text, text
    assert "tensor<4xf32>, f32" not in text, text
    assert "f32, tensor<4xf32>" not in text, text


def test_fmod_tensor_scalar_splats_scalar() -> None:
    text = make_ir(fmod_tensor_scalar_kernel, {"divisor": 3.0, "BLOCK": 4})
    print(text)
    assert "arith.constant dense<3.000000e+00> : tensor<4xf32>" in text, text
    assert "arith.divf" in text, text
    assert "arith.fptosi" in text, text
    assert "tensor<4xf32>, f32" not in text, text
    assert "f32, tensor<4xf32>" not in text, text


def test_fmod_tensor_tensor_fptosi_keeps_block_type() -> None:
    text = make_ir(fmod_tensor_tensor_kernel, {"BLOCK": 4})
    print(text)
    assert "arith.fptosi" in text, text
    assert "tensor<4xf32> to tensor<4xi32>" in text, text
    assert "tensor<4xf32> to i32" not in text, text


def test_trunc_div_rz_keeps_block_type() -> None:
    text = make_ir(trunc_div_rz_kernel, {"BLOCK": 4})
    print(text)
    assert "arith.fptosi" in text, text
    assert "tensor<4xf32> to tensor<4xi32>" in text, text
    assert "tensor<4xf32> to i32" not in text, text


if __name__ == "__main__":
    test_make_ir()
    test_pow_tensor_scalar_splats_scalar()
    test_pow_scalar_tensor_splats_scalar()
    test_fmod_tensor_scalar_splats_scalar()
    test_fmod_tensor_tensor_fptosi_keeps_block_type()
    test_trunc_div_rz_keeps_block_type()
