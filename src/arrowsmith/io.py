"""Load and validate trajectory data."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["group_id", "particle_id", "trial", "frame", "time", "x", "y"]
ROW_KEY = ["group_id", "trial", "frame", "particle_id"]


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Load a trajectory CSV and validate its structure.

    Parameters
    ----------
    path : str or Path
        Path to a CSV file with one row per particle per frame.

    Returns
    -------
    pd.DataFrame
        The validated trajectories, sorted by group, trial, frame and particle.

    Raises
    ------
    ValueError
        If required columns are missing, IDs or times contain NaNs,
        rows are duplicated, or time does not increase within a trial.
    """
    tracks = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(tracks.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    complete_columns = [c for c in REQUIRED_COLUMNS if c not in ("x", "y")]
    has_nan = tracks[complete_columns].isna().any()
    if has_nan.any():
        raise ValueError(
            f"NaNs found in columns that must be complete: "
            f"{has_nan[has_nan].index.tolist()}"
        )

    duplicated = tracks.duplicated(subset=ROW_KEY)
    if duplicated.any():
        raise ValueError(f"{duplicated.sum()} rows share the same {ROW_KEY} values")

    tracks = tracks.sort_values(ROW_KEY).reset_index(drop=True)

    dt = tracks.groupby(["group_id", "particle_id", "trial"])["time"].diff()
    if (dt <= 0).any():
        raise ValueError(
            "time must strictly increase within each group, particle and trial"
        )

    return tracks
