from triton._C import libtriton
from triton.language import core


@core.builtin
def nop(
    input,
    _semantic=None,
):
    tensor = _semantic.to_tensor(input)
    handle = libtriton.xyz.create_nop(_semantic.builder, tensor.handle)
    return core.tensor(handle, tensor.type)
