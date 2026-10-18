from unittest.mock import Mock, call

import cv2
import numpy as np
import pytest

from src.detection.models import Face
from src.detection.tracking import FaceTracker
from tests.stubs import MockCascadeClassifier


class TestFaceTracker:
    def test_init_raises_when_haarcascade_root_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.detection.tracking.cv2.data", object(), raising=False)

        with pytest.raises(RuntimeError, match="haarcascades path unavailable"):
            FaceTracker()

    def test_init_builds_cascade_classifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.detection.tracking.cv2.data",
            type("Data", (), {"haarcascades": "/opt/opencv/"})(),
            raising=False,
        )
        cascade_factory = Mock(side_effect=MockCascadeClassifier)
        monkeypatch.setattr(
            "src.detection.tracking.cv2.CascadeClassifier", cascade_factory
        )

        tracker = FaceTracker()

        cascade_factory.assert_called_once_with(
            "/opt/opencv/haarcascade_frontalface_default.xml"
        )
        assert isinstance(tracker._face_cascade, MockCascadeClassifier)
        assert (
            tracker._face_cascade.cascade_path
            == "/opt/opencv/haarcascade_frontalface_default.xml"
        )
        tracker._face_cascade.detect_multi_scale_mock.assert_not_called()

    def test_trace_detects_faces_and_draws_boxes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        frame: np.ndarray,
        grayscale: np.ndarray,
    ) -> None:
        cascade = Mock()
        cascade.detectMultiScale.return_value = [(1, 2, 3, 4), (5, 6, 7, 8)]

        tracker = FaceTracker.__new__(FaceTracker)
        tracker._face_cascade = cascade

        cvt_color = Mock(return_value=grayscale)
        rectangle = Mock()
        monkeypatch.setattr("src.detection.tracking.cv2.cvtColor", cvt_color)
        monkeypatch.setattr("src.detection.tracking.cv2.rectangle", rectangle)

        faces = tracker.trace(frame)

        assert faces == [Face(1, 2, 3, 4), Face(5, 6, 7, 8)]
        cvt_color.assert_called_once_with(frame, cv2.COLOR_BGR2GRAY)
        cascade.detectMultiScale.assert_called_once_with(
            image=grayscale,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30),
        )
        rectangle.assert_has_calls(
            [
                call(frame, (1, 2), (4, 6), FaceTracker.BOX_COLOR, 2),
                call(frame, (5, 6), (12, 14), FaceTracker.BOX_COLOR, 2),
            ]
        )
