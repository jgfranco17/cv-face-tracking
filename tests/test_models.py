from src.detection.models import Face


def test_face_area_and_location() -> None:
    face = Face(x=10, y=20, w=30, h=40)

    assert face.area() == 1200
    assert face.location() == (10, 20)
