import pytest
from forge3d._forge3d import CameraAnimation, CameraKeyframe

@pytest.mark.parametrize("time", [float("nan"),float("inf"),float("-inf"),1e300])
def test_invalid_times_are_value_errors_and_atomic(time):
    animation=CameraAnimation()
    animation.add_keyframe(1,0,45,100,60)
    with pytest.raises(ValueError,match="time"):
        CameraKeyframe(time,0,45,100,60)
    with pytest.raises(ValueError,match="time"):
        animation.add_keyframe(time,0,45,100,60)
    assert [k.time for k in animation.get_keyframes()] == [1]
    with pytest.raises((TypeError,ValueError)):
        animation.replace_keyframes([CameraKeyframe(2,0,45,100,60),object()])
    assert [k.time for k in animation.get_keyframes()] == [1]


def test_finite_keyframes_are_sorted():
    animation=CameraAnimation()
    animation.add_keyframe(3,0,45,100,60)
    animation.add_keyframe(1,0,45,100,60)
    assert [k.time for k in animation.get_keyframes()] == [1,3]
    animation.replace_keyframes([CameraKeyframe(4,0,45,100,60),CameraKeyframe(2,0,45,100,60)])
    assert [k.time for k in animation.get_keyframes()] == [2,4]
