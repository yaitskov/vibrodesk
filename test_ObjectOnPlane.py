from ObjectOnPlane import ObjectOnPlane, speed, location
import pytest


def test_speed_a0_t1():
    assert speed(0, 1) == 0


def test_speed_a1_t0():
    assert speed(1, 0) == 0


def test_speed_a2_t3():
    assert speed(2, 3) == 6


@pytest.fixture
def obj1KgR1():
    return ObjectOnPlane(m=1.0, r=0.1, g=10.0, v=5.0, x=1.0)


def test_v_and_F_codirected_and_F_more_Ftr(obj1KgR1):
    assert obj1KgR1.step(1.0, 10.0) == (10.5, 14)


def test_v_and_F_codirected_and_F_less_Ftr(obj1KgR1):
    assert obj1KgR1.step(1.0, 0.1) == (5.55, 4.1)
    assert obj1KgR1.step(100.0, 0.1) == (14.88888888888889, 0.0)


def test_v_and_F_opposed_and_F_more_Ftr(obj1KgR1):
    assert obj1KgR1.step(1.0, -9.0) == (1.25, -4)
    assert obj1KgR1.step(1.0, -3.0) == (4, 1)


def test_v_and_F_opposed_and_F_less_Ftr():
    obj = ObjectOnPlane(m=1.0, r=0.3, g=10.0, v=8.0, x=1.0)
    assert obj.step(1.0, -1) == (7, 4)
    assert obj.step(100.0, -1) == (9, 0)
