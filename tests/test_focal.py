import numpy as np
import pandas as pd
import pytest

from arrowsmith.focal import to_focal_frame


def make_frame(positions, velocities):
    """One frame of one group: a particle per (x, y) position and velocity."""
    n = len(positions)
    return pd.DataFrame(
        {
            "group_id": [0] * n,
            "trial": [0] * n,
            "frame": [0] * n,
            "particle_id": list(range(n)),
            "time": [0.0] * n,
            "x": [p[0] for p in positions],
            "y": [p[1] for p in positions],
            "vx": [v[0] for v in velocities],
            "vy": [v[1] for v in velocities],
        }
    )


def focal_row(result, focal_id):
    """The single row where the given particle is the focal."""
    rows = result[result["focal_id"] == focal_id]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize("angle", np.linspace(-np.pi, np.pi, 9))
def test_neighbour_ahead_lands_on_positive_x(angle):
    direction = (np.cos(angle), np.sin(angle))
    ahead = (2 * direction[0], 2 * direction[1])
    tracks = make_frame([(0.0, 0.0), ahead], [direction, (0.0, 0.0)])

    row = focal_row(to_focal_frame(tracks), focal_id=0)

    assert row["rel_x"] == pytest.approx(2.0)
    assert row["rel_y"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "focal_velocity, neighbour_position",
    [
        ((1.0, 0.0), (0.0, 1.0)),  # moving +x, neighbour at +y
        ((0.0, 1.0), (-1.0, 0.0)),  # moving +y, neighbour at -x
        ((-1.0, 0.0), (0.0, -1.0)),  # moving -x, neighbour at -y
    ],
)
def test_left_is_positive_y(focal_velocity, neighbour_position):
    tracks = make_frame([(0.0, 0.0), neighbour_position], [focal_velocity, (1, 0)])

    row = focal_row(to_focal_frame(tracks), focal_id=0)

    assert row["rel_x"] == pytest.approx(0.0, abs=1e-12)
    assert row["rel_y"] == pytest.approx(1.0)


def test_neighbour_velocity_is_rotated():
    # Focal moves +y; neighbour also moves +y at speed 3, i.e. straight ahead.
    tracks = make_frame([(0.0, 0.0), (1.0, 0.0)], [(0.0, 1.0), (0.0, 3.0)])

    row = focal_row(to_focal_frame(tracks), focal_id=0)

    assert row["neighbour_vx"] == pytest.approx(3.0)
    assert row["neighbour_vy"] == pytest.approx(0.0, abs=1e-12)


def test_all_ordered_pairs_without_self_pairs():
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    tracks = make_frame(positions, [(1.0, 0.0)] * 3)

    result = to_focal_frame(tracks)

    assert len(result) == 6  # 3 focals x 2 neighbours
    assert (result["focal_id"] != result["neighbour_id"]).all()


def test_stationary_focal_gives_nan():
    tracks = make_frame([(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])

    row = focal_row(to_focal_frame(tracks), focal_id=0)

    assert np.isnan(row["rel_x"])
    assert np.isnan(row["neighbour_vx"])


def test_missing_velocity_raises():
    tracks = make_frame([(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0)] * 2)

    with pytest.raises(ValueError, match="add_velocity"):
        to_focal_frame(tracks.drop(columns=["vx", "vy"]))
