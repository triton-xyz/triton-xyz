import importlib
import logging

import torch

logger = logging.getLogger(__name__)
_base = importlib.import_module("flag_gems.ops.median")


def _median_float_key_select_dim_rows(work, dim, output_shape, keepdim):
    if dim == work.ndim - 1:
        return _base._median_float_key_select_rows(work.contiguous(), output_shape)

    rows = torch.movedim(work, dim, -1).contiguous()
    row_output_shape = rows.shape[:-1]
    values, indices = _base._median_float_key_select_rows(rows, row_output_shape)
    if keepdim:
        values = torch.movedim(values.unsqueeze(-1), -1, dim)
        indices = torch.movedim(indices.unsqueeze(-1), -1, dim)
    return values, indices


_base._median_float_key_select_dim = _median_float_key_select_dim_rows


def median(inp):
    logger.debug("GEMS_XYZ MEDIAN")
    return _base.median(inp)


def median_out(inp, *, out):
    logger.debug("GEMS_XYZ MEDIAN.OUT")
    return _base.median_out(inp, out=out)


def median_dim(inp, dim=0, keepdim=False):
    logger.debug("GEMS_XYZ MEDIAN.DIM")
    return _base.median_dim(inp, dim=dim, keepdim=keepdim)


def median_dim_values(inp, dim=0, keepdim=False, *, values, indices):
    logger.debug("GEMS_XYZ MEDIAN.DIM_VALUES")
    return _base.median_dim_values(
        inp, dim=dim, keepdim=keepdim, values=values, indices=indices
    )
