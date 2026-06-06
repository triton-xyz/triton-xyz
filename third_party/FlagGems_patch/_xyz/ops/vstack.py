import logging
from typing import List, Tuple, Union

import torch

from ._concat import vstack as _vstack

logger = logging.getLogger(__name__)


def vstack(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
    logger.debug("GEMS_XYZ VSTACK")
    return _vstack(tensors)
