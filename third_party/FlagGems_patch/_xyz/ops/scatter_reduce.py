import contextlib
import importlib

_base = importlib.import_module("flag_gems.ops.scatter_reduce")


@contextlib.contextmanager
def _force_cas_for_minmax():
    old_needs_cas = _base._needs_cas_fallback
    _base._needs_cas_fallback = lambda: True
    try:
        yield
    finally:
        _base._needs_cas_fallback = old_needs_cas


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    if reduce in ("amax", "amin"):
        with _force_cas_for_minmax():
            return _base.scatter_reduce(
                inp, dim, index, src, reduce, include_self=include_self
            )
    return _base.scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    result = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(result)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    if reduce in ("amax", "amin"):
        with _force_cas_for_minmax():
            return _base.scatter_reduce_out(
                inp, dim, index, src, reduce, include_self=include_self, out=out
            )
    return _base.scatter_reduce_out(
        inp, dim, index, src, reduce, include_self=include_self, out=out
    )
