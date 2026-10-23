# arrowsmith

![tests](https://github.com/zero-neo/arrowsmith/actions/workflows/tests.yml/badge.svg)
<img width="1837" height="1534" alt="quiver" src="https://github.com/user-attachments/assets/dc9da0c5-bf0d-4242-802b-ae8919c546d3" />

From trajectories to vector fields to equations.

arrowsmith turns tracked trajectories of interacting individuals (animals,
agents, cells) into maps of how each one moves and responds relative to the
others: velocity and acceleration fields, spatial density, profiles and
direction distributions. It also simulates groups with known interaction
rules, so you can check what the maps reveal against a ground truth.
Recovering equations from the maps is the next stage of the project.

## Installation

```bash
pip install git+https://github.com/zero-neo/arrowsmith.git
```

For development, clone the repository and install it in editable mode with
the development tools:

```bash
git clone https://github.com/zero-neo/arrowsmith.git
cd arrowsmith
pip install -e ".[dev]"
pre-commit install
```

## Quick start

Simulate a pair of individuals in a circular arena and map the neighbour's
acceleration, depending on where the neighbour is relative to the focal:

```python
from arrowsmith.fields import bin_field
from arrowsmith.focal import to_focal_frame
from arrowsmith.kinematics import add_acceleration, add_velocity
from arrowsmith.plotting import plot_quiver, save_figure
from arrowsmith.simulate import simulate_zonal

tracks = simulate_zonal(duration=600, n_trials=5, seed=0)
tracks = add_acceleration(add_velocity(tracks))
focal = to_focal_frame(tracks)

field = bin_field(focal, u="neighbour_ax", v="neighbour_ay", bins=25, extent=1.0)
ax = plot_quiver(field, arrow_length=0.3)
save_figure(ax, "acceleration_map.png")
```

For your own data, replace the first line with
`tracks = load_tracks("trajectories.csv")` (from `arrowsmith.io`).

## Examples

- [`examples/simulated_force_maps.py`](examples/simulated_force_maps.py)
  simulates pairs and produces velocity, acceleration and density maps. It
  needs no data files, so it runs straight after installation.
- [`examples/force_maps.py`](examples/force_maps.py) runs the full analysis
  on a trajectory CSV, including profiles and direction distributions.

Both can be run as scripts or cell by cell (`# %%`) in an editor.

## Input data

A CSV with one row per individual per frame:

| Column        | Meaning                                              |
|---------------|------------------------------------------------------|
| `group_id`    | Which group (arena, experiment) the individual is in |
| `particle_id` | Which individual within the group                    |
| `trial`       | Recording within a group; time restarts each trial   |
| `frame`       | Frame number within the trial                        |
| `time`        | Time in seconds, strictly increasing within a trial  |
| `x`, `y`      | Position; missing values (tracking dropouts) allowed |

`load_tracks` checks this structure and raises a clear error if anything is
missing, duplicated or out of order. Simulated data from `simulate_zonal`
comes in exactly this format.

## How it works

1. **`load_tracks`** reads and validates the CSV.
2. **`add_velocity`** and **`add_acceleration`** compute velocities and
   accelerations by centred differences, separately within each individual
   and trial, using the actual time steps.
3. **`to_focal_frame`** takes each individual in turn as the *focal* and
   expresses every other individual's position, velocity and acceleration
   relative to it. The focal sits at the origin facing +x (its direction of
   travel), with +y to its left. This works for groups of any size.
4. **`bin_field`** averages a vector quantity over a grid of relative
   positions, and **`bin_density`** counts where neighbours are found. All
   arrays are indexed `[x_bin, y_bin]`.
5. **`plotting`** draws the results:
   - `plot_quiver`: a mean vector field, coloured by radial alignment
   - `plot_density`: where neighbours are found, as a heat map
   - `plot_profile`: a quantity along one axis, with standard errors
   - `plot_polar_grid`: the full distribution of directions in each region

   `save_figure` saves at 300 dpi by default, or as PDF or SVG for vector
   graphics.

### Velocity and acceleration maps

Both maps show the neighbour's motion, depending on where the neighbour is
relative to the focal, in the focal's frame of reference:

- **Neighbour velocity** (`neighbour_vx`, `neighbour_vy`): how the
  neighbour moves.
- **Neighbour acceleration** (`neighbour_ax`, `neighbour_ay`): how the
  neighbour speeds up, slows down and turns.

In `plot_quiver`, colour shows the radial alignment of each arrow: +1 points
directly away from the focal and -1 directly towards it.

The focal's own acceleration (`focal_ax`, `focal_ay`) is also available, for
mapping how the focal responds to where its neighbour is; pass those column
names to `bin_field` instead.

## Simulated data

`simulate_zonal` generates groups moving under a zonal model (after Couzin
et al. 2002) in a circular arena. Each individual swims at constant speed
and turns towards a desired direction set by three zones around it:

- **Repulsion:** move away from neighbours that are too close. This takes
  priority over the other two.
- **Orientation:** align with neighbours at intermediate distances.
- **Attraction:** move towards more distant neighbours.

Near the wall, the desired direction bends towards the wall tangent, fully
tangential at the wall itself, so individuals follow the wall the way many
fish do. With `zone_smoothness` above zero, each zone boundary becomes a
smooth ramp instead of a step, which keeps the interaction rules
continuous.

```python
from arrowsmith.simulate import ZonalParameters, simulate_zonal

params = ZonalParameters(n_individuals=2, attraction_radius=1.0, noise=1.5)
tracks = simulate_zonal(params, duration=600, n_groups=2, n_trials=5, seed=0)
```

All parameters are in arena units (radius 1) and seconds. The defaults keep
a pair together most of the time, with a distance distribution roughly like
that of larval zebrafish pairs; they are a starting point, not fitted
values. `zone_weights` returns the exact repulsion, orientation and
attraction weights at any distance, as the ground truth for checking what
the maps and recovered equations show.

## Roadmap

- Equation recovery from the maps and from the underlying dynamics,
  validated against simulated ground truth
- Nearest-neighbour selection (`neighbour_rank`) for groups larger than two
- Vector fields from images of arrows

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
