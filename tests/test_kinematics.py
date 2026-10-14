import numpy as np
import pandas as pd
import pytest

from arrowsmith.kinematics import add_velocity


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
