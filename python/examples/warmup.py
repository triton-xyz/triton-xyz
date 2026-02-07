import torch
import triton
import triton.language as tl


@triton.jit
def k(x_ptr, y_ptr, out_ptr, n, BS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    x = torch.randn(128)
    y = torch.randn(128)
    out = torch.empty_like(x)
    h = k.warmup(x, y, out, 128, BS=64, grid=(2,))
    print(type(h))
    print(h.asm.keys())


if __name__ == "__main__":
    main()
