import numpy as np
import pytest

from tests.stubs import StubTracker


@pytest.fixture
def frames() -> list[np.ndarray]:
    return [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]


@pytest.fixture
def tracker() -> StubTracker:
    return StubTracker()


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def grayscale() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.uint8)
