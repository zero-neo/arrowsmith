import pandas as pd
import pytest

from arrowsmith.io import load_tracks


def make_tracks():
    """Two particles, one group, one trial, three frames: a valid dataset."""
    return pd.DataFrame(
        {
            "group_id": [0, 0, 0, 0, 0, 0],
            "particle_id": [0, 1, 0, 1, 0, 1],
            "trial": [0, 0, 0, 0, 0, 0],
            "frame": [0, 0, 1, 1, 2, 2],
            "time": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2],
            "x": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def write(df, tmp_path):
    """Save a DataFrame to a temporary CSV and return its path."""
    path = tmp_path / "tracks.csv"
    df.to_csv(path, index=False)
    return path


def test_valid_file_loads(tmp_path):
    tracks = load_tracks(write(make_tracks(), tmp_path))

    assert tracks.shape == (6, 7)


def test_missing_column_raises(tmp_path):
    df = make_tracks().drop(columns="time")

    with pytest.raises(ValueError, match="Missing required columns"):
        load_tracks(write(df, tmp_path))


def test_nan_in_id_column_raises(tmp_path):
    df = make_tracks()
    df.loc[0, "particle_id"] = None

    with pytest.raises(ValueError, match="NaNs found"):
        load_tracks(write(df, tmp_path))


def test_duplicate_rows_raises(tmp_path):
    df = make_tracks()
    df = pd.concat([df, df.iloc[[0]]])

    with pytest.raises(ValueError, match="rows share the same"):
        load_tracks(write(df, tmp_path))


def test_non_increasing_time_raises(tmp_path):
    df = make_tracks()
    df.loc[4, "time"] = 0.05  # particle 0, frame 2: earlier than frame 1 (0.1)

    with pytest.raises(ValueError, match="time must strictly increase"):
        load_tracks(write(df, tmp_path))


def test_nan_in_xy_is_allowed(tmp_path):
    df = make_tracks()
    df.loc[2, "x"] = None

    tracks = load_tracks(write(df, tmp_path))

    assert tracks["x"].isna().sum() == 1


def test_output_is_sorted(tmp_path):
    df = make_tracks().sample(frac=1, random_state=0)  # shuffle the rows

    tracks = load_tracks(write(df, tmp_path))

    expected = tracks.sort_values(["group_id", "trial", "frame", "particle_id"])
    assert tracks.equals(expected.reset_index(drop=True))
