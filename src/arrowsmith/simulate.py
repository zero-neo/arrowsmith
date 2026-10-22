"""Simulate interacting individuals with a zonal model in a circular arena."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZonalParameters:
    """Parameters of the zonal model, in arena units and seconds.

    Each individual moves at constant speed and turns towards a desired
    direction. The desired direction comes from three zones around it,
    following Couzin et al. (2002): neighbours in the repulsion zone take
    priority and are avoided; otherwise the individual aligns with
    neighbours in the orientation zone and moves towards those in the
    attraction zone. Near the arena wall, the desired direction is blended
    with the wall tangent, fully tangential at the wall itself.

    With ``zone_smoothness`` above zero, the zone boundaries are blurred:
    each neighbour belongs to each zone with a weight between 0 and 1 that
    changes smoothly with distance, and repulsion takes over gradually
    rather than all at once. This makes the interaction rules continuous,
    which suits methods that recover equations from data.

    The defaults keep a pair together most of the time in an arena of
    radius 1, with a distance distribution roughly like that of larval
    zebrafish pairs (median distance about 0.35, mostly within 0.5) and
    frequent but not constant wall-following. They are a convenient
    starting point, not fitted values.

    Attributes
    ----------
    n_individuals : int
        Individuals per group.
    speed : float
        Constant swimming speed.
    repulsion_radius, orientation_radius, attraction_radius : float
        Outer edges of the three zones; must increase in that order.
    turning_rate : float
        Gain k in d(heading)/dt = k * sin(desired - heading), per second.
    noise : float
        Standard deviation of the heading noise, in radians per sqrt(second).
    arena_radius : float
        Radius of the circular arena, centred on the origin.
    wall_width : float
        Distance from the wall over which the tangential pull grows from 0
        (at ``wall_width`` from the wall) to 1 (at the wall).
    zone_smoothness : float
        Width of the transition at each zone boundary, as the scale of a
        logistic function. 0 gives the classic sharp zones; values around a
        tenth of the smallest zone width give a gentle blend.
    """

    n_individuals: int = 2
    speed: float = 0.3
    repulsion_radius: float = 0.2
    orientation_radius: float = 0.4
    attraction_radius: float = 1.0
    turning_rate: float = 3.0
    noise: float = 1.5
    arena_radius: float = 1.0
    wall_width: float = 0.15
    zone_smoothness: float = 0.02

    def __post_init__(self):
        """Check that the parameters describe a valid model."""
        if self.n_individuals < 1:
            raise ValueError("n_individuals must be at least 1")
        if not (
            0
            <= self.repulsion_radius
            <= self.orientation_radius
            <= self.attraction_radius
        ):
            raise ValueError(
                "Zone radii must satisfy 0 <= repulsion <= orientation <= attraction"
            )
        for name in (
            "speed",
            "turning_rate",
            "noise",
            "arena_radius",
            "wall_width",
            "zone_smoothness",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors along the last axis; zero vectors stay zero."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)


def _membership(
    distances: np.ndarray, inner: float, outer: float, smoothness: float
) -> np.ndarray:
    """Weight of membership in the band ``inner <= distance < outer``."""
    if smoothness == 0:
        return ((distances >= inner) & (distances < outer)).astype(float)

    def logistic(z):
        # Clip to avoid overflow warnings for very distant neighbours.
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    rises = logistic((distances - inner) / smoothness) if inner > -np.inf else 1.0
    falls = logistic((outer - distances) / smoothness)
    return rises * falls


def zone_weights(
    distances: np.ndarray, params: ZonalParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of the repulsion, orientation and attraction zones.

    With sharp zones (``zone_smoothness=0``) each weight is 0 or 1 and at
    most one is 1. With smooth zones they change continuously with distance.
    This is the ground truth for checking recovered interaction rules.

    Parameters
    ----------
    distances : np.ndarray
        Distances to neighbours, any shape.
    params : ZonalParameters
        Model parameters.

    Returns
    -------
    repulsion, orientation, attraction : np.ndarray
        Weights between 0 and 1, each the same shape as ``distances``.
    """
    s = params.zone_smoothness
    repulsion = _membership(distances, -np.inf, params.repulsion_radius, s)
    orientation = _membership(
        distances, params.repulsion_radius, params.orientation_radius, s
    )
    attraction = _membership(
        distances, params.orientation_radius, params.attraction_radius, s
    )
    return repulsion, orientation, attraction


def desired_direction(
    positions: np.ndarray, headings: np.ndarray, params: ZonalParameters
) -> np.ndarray:
    """Compute the unit vector each individual turns towards.

    Parameters
    ----------
    positions : np.ndarray
        Shape ``(batch, n_individuals, 2)``.
    headings : np.ndarray
        Heading angles in radians, shape ``(batch, n_individuals)``.
    params : ZonalParameters
        Model parameters.

    Returns
    -------
    np.ndarray
        Desired directions, shape ``(batch, n_individuals, 2)``.
    """
    heading_vectors = np.stack([np.cos(headings), np.sin(headings)], axis=-1)

    # Pairwise offsets: offsets[b, i, j] points from individual i to j.
    offsets = positions[:, None, :, :] - positions[:, :, None, :]
    distances = np.linalg.norm(offsets, axis=-1)
    n = positions.shape[1]
    distances[:, np.arange(n), np.arange(n)] = np.inf  # ignore self
    towards = _unit(offsets)

    repel, orient, attract = zone_weights(distances, params)

    d_repulsion = -(towards * repel[..., None]).sum(axis=2)
    # The orientation term includes the individual's own heading, so with no
    # neighbours in range it simply keeps its course.
    d_orientation = heading_vectors + (
        heading_vectors[:, None, :, :] * orient[..., None]
    ).sum(axis=2)
    d_attraction = (towards * attract[..., None]).sum(axis=2)

    # Each zone acts with the strength of its strongest member (exactly 0 or 1
    # with sharp zones). Without this, normalising a direction would give a
    # neighbour with a tiny smooth-zone weight full influence.
    repulsion_share = repel.max(axis=2, initial=0)[..., None]
    attraction_share = attract.max(axis=2, initial=0)[..., None]
    # Repulsion takes priority over orientation and attraction.
    social = _unit(
        repulsion_share * _unit(d_repulsion)
        + (1 - repulsion_share)
        * _unit(_unit(d_orientation) + attraction_share * _unit(d_attraction))
    )

    # Wall: blend towards the tangent, choosing the tangent direction closest
    # to the current heading so individuals keep their sense of rotation.
    radius = np.linalg.norm(positions, axis=-1, keepdims=True)
    radial = _unit(positions)
    tangent = np.stack([-radial[..., 1], radial[..., 0]], axis=-1)
    sign = np.where((tangent * heading_vectors).sum(axis=-1, keepdims=True) < 0, -1, 1)
    tangent = sign * tangent

    distance_to_wall = params.arena_radius - radius
    if params.wall_width > 0:
        wall_weight = np.clip(1 - distance_to_wall / params.wall_width, 0, 1)
    else:
        wall_weight = (distance_to_wall <= 0).astype(float)

    desired = _unit((1 - wall_weight) * social + wall_weight * tangent)
    # Fall back to the current heading if the blend cancels out exactly.
    return np.where(
        np.linalg.norm(desired, axis=-1, keepdims=True) > 0, desired, heading_vectors
    )


def simulate_zonal(
    params: ZonalParameters | None = None,
    duration: float = 60.0,
    fps: float = 90.0,
    n_groups: int = 1,
    n_trials: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate groups moving under the zonal model in a circular arena.

    Headings follow d(heading) = k sin(desired - heading) dt + noise dW,
    integrated with the Euler-Maruyama method at time step ``1 / fps``.
    Positions advance at constant speed along the heading; an individual
    that would leave the arena is placed back on the wall.

    Parameters
    ----------
    params : ZonalParameters, optional
        Model parameters. Defaults to ``ZonalParameters()``.
    duration : float
        Length of each trial, in seconds.
    fps : float
        Frames per second; also sets the integration time step.
    n_groups, n_trials : int
        Number of independent groups, and of trials per group. Every
        trial starts from new random positions and headings.
    seed : int, optional
        Seed for the random number generator, for reproducible output.

    Returns
    -------
    pd.DataFrame
        Trajectories in the format of ``load_tracks``: columns
        ``group_id``, ``particle_id``, ``trial``, ``frame``, ``time``,
        ``x`` and ``y``, sorted by group, trial, frame and particle.
    """
    params = ZonalParameters() if params is None else params
    rng = np.random.default_rng(seed)
    n_frames = int(round(duration * fps))
    dt = 1 / fps
    batch = n_groups * n_trials
    n = params.n_individuals

    # Uniform random start inside 80% of the arena, random headings.
    start_radius = 0.8 * params.arena_radius * np.sqrt(rng.uniform(size=(batch, n)))
    start_angle = rng.uniform(0, 2 * np.pi, size=(batch, n))
    positions = np.stack(
        [start_radius * np.cos(start_angle), start_radius * np.sin(start_angle)],
        axis=-1,
    )
    headings = rng.uniform(-np.pi, np.pi, size=(batch, n))

    trajectory = np.empty((n_frames, batch, n, 2))
    for frame in range(n_frames):
        trajectory[frame] = positions

        desired = desired_direction(positions, headings, params)
        desired_angle = np.arctan2(desired[..., 1], desired[..., 0])
        headings = (
            headings
            + params.turning_rate * np.sin(desired_angle - headings) * dt
            + params.noise * np.sqrt(dt) * rng.standard_normal((batch, n))
        )
        step = params.speed * dt
        positions = positions + step * np.stack(
            [np.cos(headings), np.sin(headings)], axis=-1
        )

        radius = np.linalg.norm(positions, axis=-1, keepdims=True)
        outside = radius > params.arena_radius
        positions = np.where(
            outside, positions / radius * params.arena_radius, positions
        )

    frames, batches, particles = np.meshgrid(
        np.arange(n_frames), np.arange(batch), np.arange(n), indexing="ij"
    )
    tracks = pd.DataFrame(
        {
            "group_id": (batches // n_trials).ravel(),
            "particle_id": particles.ravel(),
            "trial": (batches % n_trials).ravel(),
            "frame": frames.ravel(),
            "time": frames.ravel() * dt,
            "x": trajectory[..., 0].ravel(),
            "y": trajectory[..., 1].ravel(),
        }
    )
    return tracks.sort_values(
        ["group_id", "trial", "frame", "particle_id"]
    ).reset_index(drop=True)
