import numpy as np
import pytest

from backend.spatial.positioning import analyze, distance_bucket, horizontal_zone, plural

W, H = 640, 480


@pytest.mark.parametrize("cx,expected", [(0.05, "on your left"), (0.3, "slightly left"),
                                         (0.5, "ahead"), (0.7, "slightly right"), (0.95, "on your right")])
def test_horizontal_zones(cx, expected):
    assert horizontal_zone(cx)[1] == expected


def test_coarse_left_centre_right():
    assert analyze(np.array([0, 100, 100, 200]), W, H, "chair").horizontal == "left"
    assert analyze(np.array([270, 100, 370, 200]), W, H, "chair").horizontal == "centre"
    assert analyze(np.array([540, 100, 640, 200]), W, H, "chair").horizontal == "right"


def test_distance_person():
    assert distance_bucket(np.array([200, 0, 440, 470]), W, H, "person") == "near"
    assert distance_bucket(np.array([300, 100, 360, 340]), W, H, "person") == "middle"
    assert distance_bucket(np.array([310, 200, 330, 250]), W, H, "person") == "far"


def test_distance_is_class_aware():
    box = np.array([280, 180, 360, 300])  # 25% of frame height
    assert distance_bucket(box, W, H, "bottle") == "near"  # a small object this big is close
    assert distance_bucket(box, W, H, "person") == "far"  # a person this small is far


def test_in_path_and_phrase():
    sp = analyze(np.array([250, 100, 400, 470]), W, H, "chair")
    assert sp.in_path and sp.direction == "ahead" and sp.distance == "near"
    assert sp.phrase() == "ahead, near"
    assert not analyze(np.array([0, 100, 120, 300]), W, H, "chair").in_path


def test_plural():
    assert plural("person", 2) == "people" and plural("chair", 3) == "chairs" and plural("cup", 1) == "cup"
