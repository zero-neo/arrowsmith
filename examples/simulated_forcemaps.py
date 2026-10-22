"""Force maps from simulated pairs with known interaction rules.

Simulates pairs of individuals with the zonal model in a circular arena,
then runs the same analysis as ``force_maps.py``. Because the rules that
generated the data are known, this is the reference case for checking what
the maps can and cannot reveal. No data files are needed.
"""

# %% Imports and configuration
from pathlib import Path

from arrowsmith.fields import bin_density, bin_field
from arrowsmith.focal import to_focal_frame
from arrowsmith.kinematics import add_acceleration, add_velocity
from arrowsmith.plotting import plot_density, plot_quiver, save_figure
from arrowsmith.simulate import ZonalParameters, simulate_zonal

OUTPUT_DIR = Path("figures/simulated")
PARAMS = ZonalParameters(  # the defaults, written out so they are easy to change
    n_individuals=2,
    speed=0.3,
    repulsion_radius=0.2,
    orientation_radius=0.4,
    attraction_radius=1.0,
    turning_rate=3.0,
    noise=1.5,
    arena_radius=1.0,
    wall_width=0.15,
    zone_smoothness=0.02,  # 0 for classic sharp zones
)
BINS = 25
EXTENT = 1.0
MIN_COUNT = 20

# %% Simulate (2 groups x 5 trials x 10 minutes at 90 Hz)
tracks = simulate_zonal(PARAMS, duration=600, fps=90, n_groups=2, n_trials=5, seed=0)

# %% Analyse
tracks = add_acceleration(add_velocity(tracks))
focal = to_focal_frame(tracks)

field = bin_field(focal, bins=BINS, extent=EXTENT, min_count=MIN_COUNT)
acceleration = bin_field(
    focal,
    u="neighbour_ax",
    v="neighbour_ay",
    bins=BINS,
    extent=EXTENT,
    min_count=MIN_COUNT,
)
density = bin_density(focal, bins=50, extent=EXTENT)

# %% Plot and save
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_figure(plot_quiver(field, arrow_length=0.5), OUTPUT_DIR / "quiver.png")
save_figure(
    plot_quiver(acceleration, arrow_length=0.3),
    OUTPUT_DIR / "acceleration_quiver.png",
)
save_figure(plot_density(density), OUTPUT_DIR / "density.png")
print(f"Figures saved to {OUTPUT_DIR.resolve()}")
