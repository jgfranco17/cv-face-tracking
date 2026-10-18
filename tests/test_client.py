from unittest.mock import Mock, call

import cv2
import numpy as np
import pytest

from src.detection.client import FaceDetectionClient, RunSummary
from tests.stubs import StubCaptureDevice, StubTracker


@pytest.fixture
def show_func() -> Mock:
    return Mock()


class TestRunSummary:
    def test_frame_rate_uses_duration(self) -> None:
        summary = RunSummary(duration=2.0, frames_processed=5)

        assert summary.frame_rate == 2.5

    def test_frame_rate_is_zero_for_zero_duration(self) -> None:
        summary = RunSummary(duration=0.0, frames_processed=5)

        assert summary.frame_rate == 0.0


class TestFaceDetectionClient:
    def test_new_default_builds_client_with_default_dependencies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_capture = object()
        fake_tracker = object()

        video_capture = Mock(return_value=fake_capture)
        tracker_factory = Mock(return_value=fake_tracker)

        monkeypatch.setattr("src.detection.client.cv2.VideoCapture", video_capture)
        monkeypatch.setattr("src.detection.client.FaceTracker", tracker_factory)

        client = FaceDetectionClient.new_default()

        assert client._capture_device is fake_capture
        assert client._tracker is fake_tracker
        assert client._show_func is cv2.imshow
        video_capture.assert_called_once_with(0)
        tracker_factory.assert_called_once_with()

    def test_run_returns_zero_summary_when_capture_is_unavailable(
        self, tracker: StubTracker
    ) -> None:
        capture_device = StubCaptureDevice(frames=[], opened=False)
        client = FaceDetectionClient(capture_device=capture_device, tracker=tracker)

        summary = client.run()

        assert summary == RunSummary(duration=0.0, frames_processed=0)
        capture_device._is_opened.assert_called_once_with()
        capture_device._read.assert_not_called()
        capture_device._release.assert_not_called()
        tracker._trace.assert_not_called()

    def test_run_processes_frames_until_quit_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        frames: list[np.ndarray],
        tracker: StubTracker,
        show_func: Mock,
    ) -> None:
        capture_device = StubCaptureDevice(
            frames=[(True, frames[0]), (True, frames[1])]
        )
        client = FaceDetectionClient(capture_device=capture_device, tracker=tracker)
        client._show_func = show_func

        perf_counter = Mock(side_effect=[100.0, 104.0])
        wait_key = Mock(side_effect=[0, ord("q")])
        monkeypatch.setattr("src.detection.client.perf_counter", perf_counter)
        monkeypatch.setattr("src.detection.client.cv2.waitKey", wait_key)

        summary = client.run(window_name="Preview")

        assert summary == RunSummary(duration=4.0, frames_processed=2)
        capture_device._is_opened.assert_called_once_with()
        capture_device._read.assert_has_calls([call(), call()])
        assert tracker._trace.call_count == 2
        assert np.array_equal(tracker._trace.call_args_list[0].args[0], frames[0])
        assert np.array_equal(tracker._trace.call_args_list[1].args[0], frames[1])
        assert show_func.call_count == 2
        assert show_func.call_args_list[0].args[0] == "Preview"
        assert np.array_equal(show_func.call_args_list[0].args[1], frames[0])
        assert show_func.call_args_list[1].args[0] == "Preview"
        assert np.array_equal(show_func.call_args_list[1].args[1], frames[1])
        wait_key.assert_has_calls([call(1), call(1)])
        perf_counter.assert_has_calls([call(), call()])
        capture_device._release.assert_called_once_with()

    def test_run_stops_when_frame_read_fails(
        self, monkeypatch: pytest.MonkeyPatch, tracker: StubTracker
    ) -> None:
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        capture_device = StubCaptureDevice(frames=[(False, frame)])
        client = FaceDetectionClient(capture_device=capture_device, tracker=tracker)
        client._show_func = Mock()

        perf_counter = Mock(side_effect=[25.0, 26.5])
        wait_key = Mock()
        monkeypatch.setattr("src.detection.client.perf_counter", perf_counter)
        monkeypatch.setattr("src.detection.client.cv2.waitKey", wait_key)

        summary = client.run()

        assert summary == RunSummary(duration=1.5, frames_processed=0)
        capture_device._is_opened.assert_called_once_with()
        capture_device._read.assert_called_once_with()
        tracker._trace.assert_not_called()
        client._show_func.assert_not_called()
        wait_key.assert_not_called()
        perf_counter.assert_has_calls([call(), call()])
        capture_device._release.assert_called_once_with()
