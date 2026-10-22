import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from arrowsmith.fields import bin_density, bin_field
from arrowsmith.plotting import (
    angle_histograms,
    label_for,
    plot_density,
    plot_polar_grid,
    plot_profile,
    plot_quiver,
    save_figure,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def random_focal(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "rel_x": rng.uniform(-1, 1, n),
            "rel_y": rng.uniform(-1, 1, n),
            "neighbour_vx": rng.normal(1, 0.5, n),
            "neighbour_vy": rng.normal(0, 0.5, n),
        }
    )


def test_quiver_arrows_sit_at_their_own_bins():
    # One sample at (0.6, -0.6) with vector (3, 7): the arrow drawn at that
    # position must carry that vector, which fails if arrays are transposed.
    focal = pd.DataFrame(
        {"rel_x": [0.6], "rel_y": [-0.6], "neighbour_vx": [3.0], "neighbour_vy": [7.0]}
    )
    field = bin_field(focal, bins=4, extent=1.0)

    ax = plot_quiver(field, color_by=None, alpha_by_counts=False)

    quiver = ax.collections[0]
    k = np.flatnonzero(~quiver.Umask)  # arrows actually drawn
    assert len(k) == 1
    assert quiver.X[k[0]] == pytest.approx(0.75)
    assert quiver.Y[k[0]] == pytest.approx(-0.75)
    assert quiver.U[k[0]] == 3.0
    assert quiver.V[k[0]] == 7.0


@pytest.mark.parametrize("color_by", ["radial_alignment", "magnitude", None])
def test_quiver_colour_options_run(color_by):
    field = bin_field(random_focal(), bins=10, extent=1.0)

    ax = plot_quiver(field, color_by=color_by)

    assert len(ax.collections) == 1


def test_quiver_rejects_wrong_colour_shape():
    field = bin_field(random_focal(), bins=10, extent=1.0)

    with pytest.raises(ValueError, match="shape"):
        plot_quiver(field, color_by=np.zeros((3, 3)))


def test_quiver_arrow_length_sets_scale():
    field = bin_field(random_focal(), bins=10, extent=1.0)

    ax = plot_quiver(field, arrow_length=0.2, arrow_width=0.01)

    quiver = ax.collections[0]
    assert quiver.scale == pytest.approx(5.0)  # 1 / 0.2
    assert quiver.scale_units == "xy"
    assert quiver.width == 0.01


@pytest.mark.parametrize("kind", ["mean", "hexbin"])
@pytest.mark.parametrize("along", ["rel_x", "distance"])
def test_profile_kinds_run(kind, along):
    ax = plot_profile(random_focal(), along=along, kind=kind, bins=10)

    assert ax.get_xlabel() == label_for(along)


def test_profile_mean_recovers_constant():
    ax = plot_profile(random_focal(n=20_000), quantity="neighbour_vx", bins=10)

    line = ax.lines[0]
    assert np.allclose(line.get_ydata(), 1.0, atol=0.05)


def test_profile_draws_one_line_per_quantity():
    focal = random_focal(n=20_000)

    ax = plot_profile(focal, quantity=["neighbour_vx", "neighbour_vy"], bins=10)

    vx_line, vy_line = ax.lines
    assert np.allclose(vx_line.get_ydata(), 1.0, atol=0.05)
    assert np.allclose(vy_line.get_ydata(), 0.0, atol=0.05)
    assert ax.get_legend() is not None


def test_profile_hexbin_rejects_several_quantities():
    with pytest.raises(ValueError, match="single quantity"):
        plot_profile(
            random_focal(), quantity=["neighbour_vx", "neighbour_vy"], kind="hexbin"
        )


def test_angle_histograms_count_directions():
    # Every vector points along +x, so only the angle bin containing 0 fills.
    focal = random_focal().assign(neighbour_vx=1.0, neighbour_vy=0.0)

    counts, _, angle_edges = angle_histograms(focal, bins=4, angle_bins=8)

    zero_bin = np.digitize(0.0, angle_edges) - 1
    assert counts.sum() == len(focal)
    assert counts[:, :, zero_bin].sum() == len(focal)


def test_polar_grid_runs():
    ax = plot_polar_grid(random_focal(), bins=4, angle_bins=8)

    assert len(ax.collections) == 1


def test_labels_are_readable_words():
    assert label_for("rel_x") == "Relative x"
    assert label_for("some_new_column") == "Some new column"


def test_profile_default_is_forward_velocity():
    ax = plot_profile(random_focal())

    assert len(ax.lines) == 1
    assert ax.get_ylabel() == label_for("neighbour_vx")


def test_axes_have_ticks_only_at_ends_and_middle():
    field = bin_field(random_focal(), bins=10, extent=1.0)

    ax = plot_quiver(field)

    assert np.allclose(ax.get_xticks(), [-1, 0, 1])
    assert np.allclose(ax.get_yticks(), [-1, 0, 1])


def test_profile_ticks_are_round_numbers():
    ax = plot_profile(random_focal(), value_range=(-1, 1))

    assert np.allclose(ax.get_xticks(), [-1, 0, 1])
    lo, mid, hi = ax.get_yticks()
    assert (lo, hi) == ax.get_ylim()
    assert mid == pytest.approx((lo + hi) / 2)


def test_new_figures_use_high_dpi():
    ax = plot_polar_grid(random_focal(), bins=4)

    assert ax.figure.dpi == 150


def test_save_figure_writes_high_resolution_png(tmp_path):
    ax = plot_polar_grid(random_focal(), bins=4)

    path = save_figure(ax, tmp_path / "rose.png")

    with Image.open(path) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=1)


def test_density_cells_sit_at_their_own_bins():
    # Two samples at (0.6, -0.6) and none elsewhere: the coloured cell must
    # be the one at x = 0.75, y = -0.75, which fails if arrays are transposed.
    focal = pd.DataFrame({"rel_x": [0.6, 0.6], "rel_y": [-0.6, -0.6]})
    density = bin_density(focal, bins=4, extent=1.0, normalise=False)

    ax = plot_density(density)

    values = np.asarray(ax.collections[0].get_array()).reshape(4, 4)  # [y, x]
    iy, ix = np.argwhere(values > 0)[0]
    assert density.x_centres[ix] == pytest.approx(0.75)
    assert density.y_centres[iy] == pytest.approx(-0.75)


def test_acceleration_profiles_share_a_label():
    focal = random_focal().assign(focal_ax=0.0, focal_ay=0.0)

    ax = plot_profile(focal, quantity=["focal_ax", "focal_ay"], bins=10)

    assert ax.get_ylabel() == "Focal acceleration"
