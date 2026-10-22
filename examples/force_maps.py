"""Force maps from two-particle trajectories, from CSV to figures.

Runs the full arrowsmith pipeline on a trajectory CSV: load and validate,
compute velocities and accelerations, transform into the focal frame, bin
into vector fields and a density map, and plot. Run it as a script or cell
by cell (``# %%``) in an editor.

The CSV needs the columns ``group_id``, ``particle_id``, ``trial``,
``frame``, ``time``, ``x`` and ``y``; see the README for details.
"""

# %% Imports and configuration
from pathlib import Path

from arrowsmith.fields import bin_density, bin_field
from arrowsmith.focal import to_focal_frame
from arrowsmith.io import load_tracks
from arrowsmith.kinematics import add_acceleration, add_velocity
from arrowsmith.plotting import (
    plot_density,
    plot_polar_grid,
    plot_profile,
    plot_quiver,
    save_figure,
)

DATA_PATH = Path("data/private/example_pairs.csv")
OUTPUT_DIR = Path("figures")
MAX_FOCAL_SPEED = 3.0  # drop frames with tracking jumps (position units / s)
BINS = 25
EXTENT = 1.0  # bin positions in [-EXTENT, EXTENT] along both axes
MIN_COUNT = 20  # bins with fewer samples are left empty

# %% Load, compute velocities and accelerations, transform into the focal frame
tracks = load_tracks(DATA_PATH)
tracks = add_velocity(tracks)
tracks = add_acceleration(tracks)
focal = to_focal_frame(tracks)
focal = focal[focal["focal_speed"] < MAX_FOCAL_SPEED]
print(f"{len(focal):,} focal-neighbour samples")

# %% Bin into vector fields and a density map
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
OUTPUT_DIR.mkdir(exist_ok=True)

ax = plot_quiver(field, arrow_length=0.5, arrow_width=0.005)
save_figure(ax, OUTPUT_DIR / "quiver.png")

ax = plot_profile(focal, quantity="neighbour_vx", along="rel_x", value_range=(-1, 1))
save_figure(ax, OUTPUT_DIR / "profile_vx_rel_x.png")

ax = plot_profile(focal, quantity="neighbour_vy", along="rel_y", value_range=(-1, 1))
save_figure(ax, OUTPUT_DIR / "profile_vy_rel_y.png")

ax = plot_quiver(acceleration, arrow_length=0.06, arrow_width=0.005)
save_figure(ax, OUTPUT_DIR / "acceleration_quiver.png")

ax = plot_profile(focal, quantity="neighbour_ax", along="rel_x", value_range=(-1, 1))
save_figure(ax, OUTPUT_DIR / "profile_ax_rel_x.png")

ax = plot_profile(focal, quantity="neighbour_ay", along="rel_y", value_range=(-1, 1))
save_figure(ax, OUTPUT_DIR / "profile_ay_rel_y.png")

ax = plot_density(density)
save_figure(ax, OUTPUT_DIR / "density.png")

ax = plot_polar_grid(focal, bins=6, extent=EXTENT)
save_figure(ax, OUTPUT_DIR / "polar_grid.png")

print(f"Figures saved to {OUTPUT_DIR.resolve()}")
