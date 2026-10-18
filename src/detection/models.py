from dataclasses import dataclass


@dataclass(frozen=True)
class Face:
    """Data model for a detected face with bounding box coordinates."""

    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

    def location(self) -> tuple[int, int]:
        return (self.x, self.y)
