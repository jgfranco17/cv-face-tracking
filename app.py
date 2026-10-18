import logging

from src.detection.client import FaceDetectionClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s][%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def main():
    client = FaceDetectionClient.new_default()
    summary = client.run(window_name="Face Tracking Test")
    logger.info(
        f"Ran face detection for {summary.duration:.2f}s ({summary.frames_processed} frames)"
    )
    logger.info(f"Average frame rate: {summary.frame_rate:.2f} fps")


if __name__ == "__main__":
    main()
