from typing import cast
from unittest.mock import Mock

import cv2
import numpy as np

from src.detection.models import Face
from src.detection.tracking import TrackerBase


class StubCaptureDevice:
    def __init__(
        self, frames: list[tuple[bool, np.ndarray]], opened: bool = True
    ) -> None:
        self._is_opened = Mock(return_value=opened)
        self._read = Mock(side_effect=frames)
        self._release = Mock()

    def isOpened(self) -> bool:
        return cast(bool, self._is_opened())

    def read(self) -> tuple[bool, np.ndarray]:
        return cast(tuple[bool, np.ndarray], self._read())

    def release(self) -> None:
        self._release()


class StubTracker(TrackerBase):
    def __init__(self) -> None:
        self._trace = Mock(return_value=[])

    def trace(self, frame: cv2.typing.MatLike) -> list[Face]:
        return cast(list[Face], self._trace(frame))


class MockCascadeClassifier:
    def __init__(self, cascade_path: str) -> None:
        self.cascade_path = cascade_path
        self.detect_multi_scale_mock = Mock(return_value=[])

    def detectMultiScale(
        self,
        *,
        image: np.ndarray,
        scaleFactor: float,
        minNeighbors: int,
        minSize: tuple[int, int],
    ) -> list[tuple[int, int, int, int]]:
        return cast(
            list[tuple[int, int, int, int]],
            self.detect_multi_scale_mock(
                image=image,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
            ),
        )
