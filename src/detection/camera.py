import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class CaptureDevice(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray]: ...

    def release(self) -> None: ...
