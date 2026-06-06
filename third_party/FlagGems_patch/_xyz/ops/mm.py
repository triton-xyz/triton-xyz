import logging

import torch

from flag_gems.ops.mm import (
    cluster_remote_mm,
    cluster_remote_mm_scenario,
    general_mm,
    get_higher_dtype,
    streamk_mm,
    streamk_scenario,
)
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)


def mm(a, b):
    logger.debug("GEMS_XYZ MM")

    device = a.device
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)

    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    if cluster_remote_mm_scenario(a, b, c, M, N, K):
        return cluster_remote_mm(a, b, c, M, N, K)
    return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    logger.debug("GEMS_XYZ MM_OUT")

    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    if cluster_remote_mm_scenario(a, b, out, M, N, K):
        return cluster_remote_mm(a, b, out, M, N, K)
    return general_mm(a, b, out, M, N, K)
