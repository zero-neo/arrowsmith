import numpy as np
import pandas as pd
import pytest

from arrowsmith.fields import (
    VectorField,
    bin_density,
    bin_field,
    divergence,
    radial_alignment,
)


def make_focal(rel_x, rel_y, vx, vy):
    return pd.DataFrame(
        {"rel_x": rel_x, "rel_y": rel_y, "neighbour_vx": vx, "neighbour_vy": vy}
    )


def analytic_field(u_func, v_func, n=20, half_width=1.0):
    """A VectorField with u and v given by functions of position.

    An even number of bins keeps every bin centre away from the origin.
    """
    edges = np.linspace(-half_width, half_width, n + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    x, y = np.meshgrid(centres, centres, indexing="ij")
    counts = np.ones_like(x)
    return VectorField(edges, edges, u_func(x, y), v_func(x, y), counts)


def test_uniform_field_has_uniform_mean():
    rng = np.random.default_rng(0)
    n = 10_000
    focal = make_focal(
        rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), [1.0] * n, [0.0] * n
    )

    field = bin_field(focal, bins=10, extent=1.0)

    assert np.allclose(field.u, 1.0)
    assert np.allclose(field.v, 0.0)
    assert field.counts.sum() == n


def test_arrays_are_indexed_x_then_y():
    # One sample at x = 0.6, y = -0.6: the first index must locate x.
    focal = make_focal([0.6], [-0.6], [3.0], [7.0])

    field = bin_field(focal, bins=4, extent=1.0)

    ix, iy = np.argwhere(field.counts == 1)[0]
    assert field.x_centres[ix] == pytest.approx(0.75)
    assert field.y_centres[iy] == pytest.approx(-0.75)
    assert field.u[ix, iy] == 3.0
    assert field.v[ix, iy] == 7.0


def test_bins_below_min_count_are_nan():
    focal = make_focal([0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [1.0] * 3, [0.0] * 3)

    field = bin_field(focal, bins=2, extent=1.0, min_count=2)

    assert field.u[1, 1] == 1.0  # two samples
    assert np.isnan(field.u[0, 0])  # one sample
    assert np.isnan(field.u[0, 1])  # empty


def test_rows_with_nan_are_ignored():
    focal = make_focal([0.5, 0.5], [0.5, 0.5], [1.0, np.nan], [0.0, 0.0])

    field = bin_field(focal, bins=2, extent=1.0)

    assert field.counts.sum() == 1
    assert field.u[1, 1] == 1.0


def test_missing_column_raises():
    focal = make_focal([0.5], [0.5], [1.0], [0.0]).drop(columns="neighbour_vy")

    with pytest.raises(ValueError, match="Missing columns"):
        bin_field(focal)


@pytest.mark.parametrize(
    "u_func, v_func, expected",
    [
        (lambda x, y: x, lambda x, y: y, 1.0),  # outward
        (lambda x, y: -x, lambda x, y: -y, -1.0),  # inward
        (lambda x, y: -y, lambda x, y: x, 0.0),  # rotation
    ],
)
def test_radial_alignment_simple_fields(u_func, v_func, expected):
    field = analytic_field(u_func, v_func)

    alignment = radial_alignment(field)

    assert np.allclose(alignment, expected, atol=1e-12)


def test_radial_alignment_asymmetric_field():
    # v = (x, 2y) is not symmetric under swapping x and y, so a transposed
    # array (the bug in the original code) gives different values.
    field = analytic_field(lambda x, y: x, lambda x, y: 2 * y)
    x, y = field.mesh()
    expected = (x**2 + 2 * y**2) / (np.hypot(x, y) * np.hypot(x, 2 * y))

    assert np.allclose(radial_alignment(field), expected)


def test_divergence_of_linear_field():
    # div(x, 2y) = 1 + 2 = 3; swapped axes would give 0.
    field = analytic_field(lambda x, y: x, lambda x, y: 2 * y)

    assert np.allclose(divergence(field), 3.0)


def test_density_integrates_to_one():
    rng = np.random.default_rng(0)
    focal = make_focal(rng.normal(0, 0.3, 5_000), rng.normal(0, 0.3, 5_000), 0, 0)

    density = bin_density(focal, bins=20, extent=1.0)

    bin_area = np.outer(np.diff(density.x_edges), np.diff(density.y_edges))
    assert (density.values * bin_area).sum() == pytest.approx(1.0)


def test_density_counts_and_indexing():
    focal = make_focal([0.6, 0.6, -0.6], [-0.6, -0.6, 0.6], 0, 0)

    density = bin_density(focal, bins=4, extent=1.0, normalise=False)

    assert density.values[3, 0] == 2  # x = 0.75, y = -0.75
    assert density.values[0, 3] == 1  # x = -0.75, y = 0.75
    assert density.counts.sum() == 3


def test_density_missing_column_raises():
    focal = make_focal([0.5], [0.5], 0, 0).drop(columns="rel_y")

    with pytest.raises(ValueError, match="Missing columns"):
        bin_density(focal)
