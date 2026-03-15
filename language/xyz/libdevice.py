from triton._C import libtriton
from triton.language import core
from triton.language.semantic import TritonSemantic


@core.builtin
def nop(
    input,
    _semantic: TritonSemantic = None,  # ty:ignore
):
    tensor = _semantic.to_tensor(input)
    handle = libtriton.xyz.create_nop(_semantic.builder, tensor.handle)  # ty:ignore
    return core.tensor(handle, tensor.type)
