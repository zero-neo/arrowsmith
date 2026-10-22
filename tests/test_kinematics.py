import numpy as np
import pandas as pd
import pytest

from arrowsmith.kinematics import add_acceleration, add_velocity


def make_track(x, time):
    """One particle in one group and trial, moving along x."""
    n = len(x)
    return pd.DataFrame(
        {
            "group_id": [0] * n,
            "particle_id": [0] * n,
            "trial": [0] * n,
            "frame": list(range(n)),
            "time": time,
            "x": x,
            "y": [0.0] * n,
        }
    )


def test_constant_velocity():
    time = np.arange(5) * 0.1  # 0.0, 0.1, 0.2, 0.3, 0.4 seconds
    tracks = make_track(x=2 * time, time=time)  # moves at 2 units per second

    result = add_velocity(tracks)

    assert np.isnan(result["vx"].iloc[0])  # first frame: no previous frame
    assert np.isnan(result["vx"].iloc[-1])  # last frame: no next frame
    assert result["vx"].iloc[1:-1].tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_input_not_modified():
    time = np.arange(5) * 0.1
    tracks = make_track(x=2 * time, time=time)

    add_velocity(tracks)

    assert "vx" not in tracks.columns


def test_constant_acceleration():
    time = np.arange(7) * 0.1
    tracks = add_velocity(make_track(x=time**2, time=time))  # a = 2

    result = add_acceleration(tracks)

    assert result["ax"].iloc[:2].isna().all()  # first two frames
    assert result["ax"].iloc[-2:].isna().all()  # last two frames
    assert result["ax"].iloc[2:-2].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert result["ay"].iloc[2:-2].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_acceleration_does_not_modify_input():
    time = np.arange(7) * 0.1
    tracks = add_velocity(make_track(x=time**2, time=time))

    add_acceleration(tracks)

    assert "ax" not in tracks.columns


def test_acceleration_needs_velocity():
    time = np.arange(7) * 0.1

    with pytest.raises(ValueError, match="add_velocity"):
        add_acceleration(make_track(x=time, time=time))
