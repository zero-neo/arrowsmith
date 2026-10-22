import numpy as np
import pytest

from arrowsmith.io import load_tracks
from arrowsmith.kinematics import add_velocity
from arrowsmith.simulate import (
    ZonalParameters,
    desired_direction,
    simulate_zonal,
    zone_weights,
)

# Classic sharp zones at fixed radii, so tests don't depend on the defaults.
CLASSIC = {
    "repulsion_radius": 0.05,
    "orientation_radius": 0.15,
    "attraction_radius": 0.4,
    "zone_smoothness": 0.0,
}

# Interactions switched off, for testing one ingredient at a time.
NO_SOCIAL = {
    "repulsion_radius": 0.0,
    "orientation_radius": 0.0,
    "attraction_radius": 0.0,
    "zone_smoothness": 0.0,
}


def heading_vectors(angles):
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def test_output_passes_load_tracks(tmp_path):
    tracks = simulate_zonal(duration=1.0, n_groups=2, n_trials=3, seed=0)
    path = tmp_path / "sim.csv"
    tracks.to_csv(path, index=False)

    loaded = load_tracks(path)

    assert len(loaded) == 2 * 3 * 2 * 90  # groups x trials x fish x frames
    assert set(loaded["group_id"]) == {0, 1}
    assert set(loaded["trial"]) == {0, 1, 2}


def test_same_seed_gives_same_output():
    first = simulate_zonal(duration=1.0, seed=1)
    second = simulate_zonal(duration=1.0, seed=1)
    other = simulate_zonal(duration=1.0, seed=2)

    assert first.equals(second)
    assert not first.equals(other)


def test_individuals_stay_inside_the_arena():
    tracks = simulate_zonal(duration=20.0, n_trials=4, seed=0)

    assert np.hypot(tracks["x"], tracks["y"]).max() <= 1.0 + 1e-9


def test_speed_is_constant():
    params = ZonalParameters(speed=0.3, noise=0.0)
    tracks = add_velocity(simulate_zonal(params, duration=5.0, seed=0))

    speed = tracks["speed"].dropna()
    assert speed.mean() == pytest.approx(0.3, rel=0.01)


def test_more_individuals_per_group():
    tracks = simulate_zonal(ZonalParameters(n_individuals=5), duration=1.0, seed=0)

    assert tracks["particle_id"].nunique() == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repulsion_radius": 0.5, "orientation_radius": 0.3},
        {"speed": -1.0},
        {"n_individuals": 0},
        {"zone_smoothness": -0.01},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        ZonalParameters(**kwargs)


def test_alone_far_from_wall_keeps_course():
    params = ZonalParameters(**NO_SOCIAL)
    positions = np.array([[[0.0, 0.0]]])
    headings = np.array([[0.7]])

    desired = desired_direction(positions, headings, params)

    assert np.allclose(desired, heading_vectors(headings))


def test_at_the_wall_desired_direction_is_tangential():
    # At the wall on the +x side, heading mostly outwards and slightly up.
    params = ZonalParameters(**NO_SOCIAL)
    positions = np.array([[[1.0, 0.0]]])
    headings = np.array([[0.2]])

    desired = desired_direction(positions, headings, params)

    assert np.allclose(desired, [[[0.0, 1.0]]])  # tangent, same rotation sense


def test_repulsion_points_away_from_close_neighbour():
    params = ZonalParameters(**CLASSIC)
    positions = np.array([[[0.0, 0.0], [0.02, 0.0]]])  # inside repulsion zone
    headings = np.array([[np.pi / 2, np.pi / 2]])

    desired = desired_direction(positions, headings, params)

    assert np.allclose(desired[0, 0], [-1.0, 0.0])
    assert np.allclose(desired[0, 1], [1.0, 0.0])


def test_orientation_aligns_with_neighbour():
    params = ZonalParameters(**CLASSIC)
    positions = np.array([[[0.0, 0.0], [0.1, 0.0]]])  # orientation zone
    headings = np.array([[0.0, np.pi / 2]])

    desired = desired_direction(positions, headings, params)

    # Focal 0 turns halfway between its own heading and its neighbour's.
    expected = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(desired[0, 0], expected)


def test_attraction_pulls_pair_together():
    params = ZonalParameters(**{**CLASSIC, "noise": 0.0, "orientation_radius": 0.05})
    positions = np.array([[[0.0, 0.0], [0.3, 0.0]]])  # attraction zone
    headings = np.array([[np.pi / 2, np.pi / 2]])

    desired = desired_direction(positions, headings, params)

    assert desired[0, 0, 0] > 0  # focal 0 bends towards +x, where 1 is
    assert desired[0, 1, 0] < 0  # and 1 towards 0


def test_sharp_zone_weights_are_indicators():
    params = ZonalParameters(**CLASSIC)  # zones at 0.05, 0.15, 0.4
    distances = np.array([0.02, 0.1, 0.3, 0.6])

    repulsion, orientation, attraction = zone_weights(distances, params)

    assert repulsion.tolist() == [1, 0, 0, 0]
    assert orientation.tolist() == [0, 1, 0, 0]
    assert attraction.tolist() == [0, 0, 1, 0]


def test_smooth_zone_weights_are_continuous():
    params = ZonalParameters(**{**CLASSIC, "zone_smoothness": 0.01})
    distances = np.linspace(0, 0.6, 6001)  # steps of 0.0001

    for weights in zone_weights(distances, params):
        assert ((weights >= 0) & (weights <= 1)).all()
        assert np.abs(np.diff(weights)).max() < 0.01  # no jumps


def test_smooth_zones_match_sharp_zones_away_from_boundaries():
    sharp = ZonalParameters(**CLASSIC)
    smooth = ZonalParameters(**{**CLASSIC, "zone_smoothness": 0.002})
    distances = np.array([0.02, 0.1, 0.3, 0.6])  # all far from a boundary

    for w_sharp, w_smooth in zip(
        zone_weights(distances, sharp), zone_weights(distances, smooth), strict=True
    ):
        assert np.allclose(w_sharp, w_smooth, atol=1e-3)


def test_smooth_repulsion_takes_over_gradually():
    # A neighbour exactly at the repulsion boundary is half repulsion, so the
    # desired direction lies between "away" and the orientation/attraction
    # direction rather than snapping to either.
    params = ZonalParameters(**{**CLASSIC, "zone_smoothness": 0.01, "wall_width": 0.0})
    positions = np.array([[[0.0, 0.0], [0.05, 0.0]]])
    headings = np.array([[np.pi / 2, np.pi / 2]])

    desired = desired_direction(positions, headings, params)

    x, y = desired[0, 0]
    assert -1 < x < 0  # partly away from the neighbour (at +x)
    assert y > 0  # partly along the shared heading


def test_smooth_simulation_runs(tmp_path):
    params = ZonalParameters(zone_smoothness=0.01)
    tracks = simulate_zonal(params, duration=2.0, seed=0)
    path = tmp_path / "sim.csv"
    tracks.to_csv(path, index=False)

    assert len(load_tracks(path)) == 2 * 180


def test_smooth_attraction_does_not_reach_beyond_its_zone():
    # A neighbour well outside the attraction zone has a tiny smooth weight;
    # it must have a tiny effect, not full attraction.
    params = ZonalParameters(**{**CLASSIC, "zone_smoothness": 0.01, "wall_width": 0.0})
    positions = np.array([[[0.0, 0.0], [0.7, 0.0]]])
    headings = np.array([[np.pi / 2, np.pi / 2]])

    desired = desired_direction(positions, headings, params)

    assert np.allclose(desired[0, 0], [0.0, 1.0], atol=1e-6)  # keeps course


def test_default_pair_stays_together():
    # The defaults are meant to keep a pair interacting most of the time.
    tracks = simulate_zonal(duration=60.0, n_trials=4, seed=0)
    first = tracks[tracks["particle_id"] == 0][["x", "y"]].to_numpy()
    second = tracks[tracks["particle_id"] == 1][["x", "y"]].to_numpy()

    distance = np.hypot(*(first - second).T)

    assert np.median(distance) < 0.5
