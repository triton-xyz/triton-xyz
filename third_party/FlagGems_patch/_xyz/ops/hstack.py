import logging
from typing import List, Tuple, Union

import torch

from ._concat import hstack as _hstack

logger = logging.getLogger(__name__)


def hstack(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
    logger.debug("GEMS_XYZ HSTACK")
    return _hstack(tensors)
