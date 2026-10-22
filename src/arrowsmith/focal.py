"""Transform neighbour positions and velocities into the focal frame."""

import numpy as np
import pandas as pd

FRAME_COLUMNS = ["group_id", "trial", "frame"]
REQUIRED_COLUMNS = [*FRAME_COLUMNS, "particle_id", "time", "x", "y", "vx", "vy"]


def _rotate(
    x: pd.Series, y: pd.Series, cos: pd.Series, sin: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Rotate vectors by minus the heading angle, into the focal frame."""
    return x * cos + y * sin, -x * sin + y * cos


def to_focal_frame(tracks: pd.DataFrame) -> pd.DataFrame:
    """Express each neighbour's position and velocity in each focal's frame.

    For every frame, each particle is taken in turn as the focal particle,
    and every other particle in the same group is a neighbour. The focal
    sits at the origin facing +x (its direction of travel), with +y to its
    left. The focal's heading is undefined when its velocity is NaN or
    exactly zero; those rows get NaN in all transformed columns.

    If ``tracks`` has acceleration columns (from ``add_acceleration``),
    both individuals' accelerations are rotated into the focal frame too:
    ``neighbour_ax`` and ``neighbour_ay`` for the neighbour, and
    ``focal_ax`` (speeding up or slowing down) and ``focal_ay`` (turning
    towards +y) for the focal itself.

    Parameters
    ----------
    tracks : pd.DataFrame
        Trajectories with velocity columns, as returned by ``add_velocity``.

    Returns
    -------
    pd.DataFrame
        One row per focal, neighbour and frame, with columns ``group_id``,
        ``trial``, ``frame``, ``time``, ``focal_id``, ``neighbour_id``,
        ``focal_speed``, ``rel_x`` and ``rel_y`` (neighbour position), and
        ``neighbour_vx`` and ``neighbour_vy`` (neighbour velocity), all in
        the focal frame. With accelerations in the input, also
        ``focal_ax``, ``focal_ay``, ``neighbour_ax`` and ``neighbour_ay``.

    Raises
    ------
    ValueError
        If required columns, including the velocity columns, are missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(tracks.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            "Velocities come from add_velocity()."
        )

    has_acceleration = {"ax", "ay"} <= set(tracks.columns)
    columns = ["particle_id", "time", "x", "y", "vx", "vy"]
    if has_acceleration:
        columns += ["ax", "ay"]
    pairs = tracks[FRAME_COLUMNS + columns].merge(
        tracks[FRAME_COLUMNS + columns],
        on=FRAME_COLUMNS,
        suffixes=("_focal", "_neighbour"),
    )
    pairs = pairs[pairs["particle_id_focal"] != pairs["particle_id_neighbour"]]

    focal_speed = np.hypot(pairs["vx_focal"], pairs["vy_focal"])
    heading = np.arctan2(pairs["vy_focal"], pairs["vx_focal"]).where(focal_speed > 0)
    cos, sin = np.cos(heading), np.sin(heading)

    dx = pairs["x_neighbour"] - pairs["x_focal"]
    dy = pairs["y_neighbour"] - pairs["y_focal"]
    rel_x, rel_y = _rotate(dx, dy, cos, sin)
    neighbour_vx, neighbour_vy = _rotate(
        pairs["vx_neighbour"], pairs["vy_neighbour"], cos, sin
    )

    focal = pd.DataFrame(
        {
            "group_id": pairs["group_id"],
            "trial": pairs["trial"],
            "frame": pairs["frame"],
            "time": pairs["time_focal"],
            "focal_id": pairs["particle_id_focal"],
            "neighbour_id": pairs["particle_id_neighbour"],
            "focal_speed": focal_speed,
            "rel_x": rel_x,
            "rel_y": rel_y,
            "neighbour_vx": neighbour_vx,
            "neighbour_vy": neighbour_vy,
        }
    )
    if has_acceleration:
        for who in ("focal", "neighbour"):
            ax, ay = _rotate(pairs[f"ax_{who}"], pairs[f"ay_{who}"], cos, sin)
            focal[f"{who}_ax"] = ax
            focal[f"{who}_ay"] = ay
    return focal.reset_index(drop=True)
