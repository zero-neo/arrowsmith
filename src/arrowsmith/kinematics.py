"""Compute velocities from trajectories."""

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
    tracks_prev = tracks.groupby(TRACK_COLUMNS).shift(1)
    tracks_next = tracks.groupby(TRACK_COLUMNS).shift(-1)

    dt = tracks_next["time"] - tracks_prev["time"]
    tracks["vx"] = (tracks_next["x"] - tracks_prev["x"]) / dt

    tracks["vy"] = (tracks_next["y"] - tracks_prev["y"]) / dt
    tracks["speed"] = np.sqrt(tracks["vx"] ** 2 + tracks["vy"] ** 2)

    return tracks
