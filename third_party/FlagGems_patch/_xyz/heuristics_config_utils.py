import os

import torch
import triton

from flag_gems.runtime.backend._nvidia import heuristics_config_utils as nvidia_hcu


_MAX_TILE_N_PER_ROW = nvidia_hcu._MAX_TILE_N_PER_ROW
_MIN_TILE_N = nvidia_hcu._MIN_TILE_N
_MAX_ONE_TILE_N = nvidia_hcu._MAX_ONE_TILE_N


def _num_sms():
    return max(1, os.cpu_count() or 1)


def argmax_heur_tile_k(args):
    max_tile_k = 512
    num_sms = _num_sms()

    k = args["K"]
    m = args["M"]
    dtype = "fp32" if args["inp"].dtype == torch.float32 else "fp16"

    if m == 64 and k == 512:
        return 64 if dtype == "fp32" else 128

    if k <= 128:
        return 1 << (k.bit_length() - 1) if k > 0 else 1

    tile_k = 64
    upper_bound = min(k, max_tile_k)

    while tile_k <= upper_bound:
        num_blocks = m * triton.cdiv(k, tile_k)
        num_waves = num_blocks / num_sms

        if num_waves > 1 and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break

    return tile_k


def softmax_heur_tile_k(args):
    max_tile_k = 8192
    num_sms = _num_sms()
    tile_k = 1
    upper_bound = min(args["K"], max_tile_k)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / num_sms
        if num_waves > 1 and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def mean_heur_tile_k(args):
    max_tile_k = 512
    max_grid_y = 65535
    num_sms = _num_sms()
    tile_k = 1
    upper_bound = min(args["K"], max_tile_k)
    max_tile_k_allowed_by_tile_n = max(1, _MAX_TILE_N_PER_ROW // _MIN_TILE_N)
    upper_bound = min(upper_bound, max_tile_k_allowed_by_tile_n)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / num_sms
        if num_waves > 1 and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break

    min_tile_k = triton.cdiv(args["K"], max_grid_y)
    if min_tile_k > tile_k:
        tile_k = triton.next_power_of_2(min_tile_k)
    return tile_k


def argmin_heur_block_m(args):
    return 1


HEURISTICS_CONFIGS = dict(nvidia_hcu.HEURISTICS_CONFIGS)
HEURISTICS_CONFIGS["argmin"] = {
    **HEURISTICS_CONFIGS["argmin"],
    "BLOCK_M": argmin_heur_block_m,
}
HEURISTICS_CONFIGS["argmax_non_inner"] = {
    **HEURISTICS_CONFIGS["argmax_non_inner"],
    "TILE_K": argmax_heur_tile_k,
}
HEURISTICS_CONFIGS["softmax_non_inner"] = {
    **HEURISTICS_CONFIGS["softmax_non_inner"],
    "TILE_K": softmax_heur_tile_k,
}
HEURISTICS_CONFIGS["mean_non_inner"] = {
    **HEURISTICS_CONFIGS["mean_non_inner"],
    "TILE_K": mean_heur_tile_k,
}
