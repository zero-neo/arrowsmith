"""Compute velocities and accelerations from trajectories."""

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["group_id", "particle_id", "trial"]


def add_velocity(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add velocity columns computed by centred differences.

    Velocity at frame t is (x[t+1] - x[t-1]) / (time[t+1] - time[t-1]),
    computed separately within each group, particle and trial. The first
    and last frame of each track, and frames next to missing positions,
    get NaN.

    Parameters
    ----------
    tracks : pd.DataFrame
        Validated trajectories, as returned by ``load_tracks``.

    Returns
    -------
    pd.DataFrame
        A copy of ``tracks`` with added columns ``vx``, ``vy`` and ``speed``.
    """
    tracks = tracks.copy()
    tracks_prev = tracks.groupby(TRACK_COLUMNS)[["x", "y", "time"]].shift(1)
    tracks_next = tracks.groupby(TRACK_COLUMNS)[["x", "y", "time"]].shift(-1)

    dt = tracks_next["time"] - tracks_prev["time"]
    tracks["vx"] = (tracks_next["x"] - tracks_prev["x"]) / dt
    tracks["vy"] = (tracks_next["y"] - tracks_prev["y"]) / dt
    tracks["speed"] = np.sqrt(tracks["vx"] ** 2 + tracks["vy"] ** 2)

    return tracks


def add_acceleration(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add acceleration columns computed by centred differences of velocity.

    Acceleration at frame t is (vx[t+1] - vx[t-1]) / (time[t+1] - time[t-1]),
    and likewise for y, computed separately within each group, particle and
    trial. Because velocity is itself a centred difference, the first and
    last two frames of each track get NaN, as do frames within two of a
    missing position.

    Differencing twice amplifies tracking noise strongly, so individual
    accelerations are noisy; averages over many frames, as in binned maps,
    are much more reliable.

    Parameters
    ----------
    tracks : pd.DataFrame
        Trajectories with velocity columns, as returned by ``add_velocity``.

    Returns
    -------
    pd.DataFrame
        A copy of ``tracks`` with added columns ``ax`` and ``ay``.

    Raises
    ------
    ValueError
        If the velocity columns are missing.
    """
    missing = {"vx", "vy"} - set(tracks.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. Run add_velocity() first."
        )

    tracks = tracks.copy()
    grouped = tracks.groupby(TRACK_COLUMNS)[["vx", "vy", "time"]]
    prev = grouped.shift(1)
    next_ = grouped.shift(-1)

    dt = next_["time"] - prev["time"]
    tracks["ax"] = (next_["vx"] - prev["vx"]) / dt
    tracks["ay"] = (next_["vy"] - prev["vy"]) / dt

    return tracks
