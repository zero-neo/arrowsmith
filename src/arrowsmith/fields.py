"""Bin focal-frame data into vector fields and compute field properties."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VectorField:
    """Mean vector field on a regular grid.

    All 2D arrays are indexed ``[x_bin, y_bin]``: the first axis runs along
    x and the second along y. When plotting against a mesh made with
    ``np.meshgrid(x_centres, y_centres)`` (the default ``"xy"`` indexing),
    transpose them first.

    Attributes
    ----------
    x_edges, y_edges : np.ndarray
        Bin edges along x and y.
    u, v : np.ndarray
        Mean x and y components in each bin. NaN where a bin has fewer
        samples than the ``min_count`` used to build the field.
    counts : np.ndarray
        Number of samples in each bin.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    u: np.ndarray
    v: np.ndarray
    counts: np.ndarray

    @property
    def x_centres(self) -> np.ndarray:
        """Bin centres along x."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centres(self) -> np.ndarray:
        """Bin centres along y."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def magnitude(self) -> np.ndarray:
        """Length of the mean vector in each bin."""
        return np.hypot(self.u, self.v)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Bin-centre coordinates as 2D arrays indexed ``[x_bin, y_bin]``."""
        return np.meshgrid(self.x_centres, self.y_centres, indexing="ij")


@dataclass(frozen=True)
class ScalarField:
    """A single value per bin on a regular grid, such as a density.

    ``values`` and ``counts`` are indexed ``[x_bin, y_bin]``, as in
    ``VectorField``.

    Attributes
    ----------
    x_edges, y_edges : np.ndarray
        Bin edges along x and y.
    values : np.ndarray
        The value in each bin.
    counts : np.ndarray
        Number of samples in each bin.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    values: np.ndarray
    counts: np.ndarray

    @property
    def x_centres(self) -> np.ndarray:
        """Bin centres along x."""
        return (self.x_edges[:-1] + self.x_edges[1:]) / 2

    @property
    def y_centres(self) -> np.ndarray:
        """Bin centres along y."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2


Extent = float | tuple[tuple[float, float], tuple[float, float]] | None


def _resolve_extent(
    positions: pd.DataFrame, extent: Extent
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Turn the ``extent`` argument into ``((x_min, x_max), (y_min, y_max))``."""
    if extent is None:
        extent = float(np.max(np.abs(positions.to_numpy())))
    if np.isscalar(extent):
        return ((-extent, extent), (-extent, extent))
    return extent


def bin_field(
    focal: pd.DataFrame,
    u: str = "neighbour_vx",
    v: str = "neighbour_vy",
    x: str = "rel_x",
    y: str = "rel_y",
    bins: int | tuple[int, int] = 50,
    extent: Extent = None,
    min_count: int = 1,
) -> VectorField:
    """Average a vector quantity over a grid of positions.

    Parameters
    ----------
    focal : pd.DataFrame
        Focal-frame data, as returned by ``to_focal_frame``. Rows with NaN
        in any of the four columns used are ignored.
    u, v : str
        Columns holding the x and y components of the vector to average.
    x, y : str
        Columns holding the positions to bin over.
    bins : int or tuple of int
        Number of bins along each axis, or ``(n_x, n_y)``.
    extent : float or tuple, optional
        Region to bin. A float ``a`` means ``[-a, a]`` along both axes; a
        tuple gives ``((x_min, x_max), (y_min, y_max))``. Defaults to a
        symmetric square covering all positions.
    min_count : int
        Bins with fewer samples than this get NaN in ``u`` and ``v``.

    Returns
    -------
    VectorField
        The binned mean field and sample counts.

    Raises
    ------
    ValueError
        If any of the named columns are missing.
    """
    missing = {u, v, x, y} - set(focal.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = focal[[x, y, u, v]].dropna()
    extent = _resolve_extent(data[[x, y]], extent)

    def histogram(weights=None):
        return np.histogram2d(
            data[x], data[y], bins=bins, range=extent, weights=weights
        )

    counts, x_edges, y_edges = histogram()
    u_sum, _, _ = histogram(data[u])
    v_sum, _, _ = histogram(data[v])

    enough = counts >= max(min_count, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_mean = np.where(enough, u_sum / counts, np.nan)
        v_mean = np.where(enough, v_sum / counts, np.nan)

    return VectorField(x_edges, y_edges, u_mean, v_mean, counts)


def radial_alignment(field: VectorField) -> np.ndarray:
    """Cosine between the mean vector and the outward radial direction.

    +1 means the mean vector points directly away from the origin (the
    focal), -1 directly towards it, and 0 means tangential motion. NaN
    where the mean vector is NaN or zero, or the bin centre is the origin.

    Parameters
    ----------
    field : VectorField
        The binned field.

    Returns
    -------
    np.ndarray
        Alignment per bin, indexed ``[x_bin, y_bin]`` like ``field.u``.
    """
    x, y = field.mesh()
    r = np.hypot(x, y)
    norm = field.magnitude * r
    with np.errstate(invalid="ignore", divide="ignore"):
        alignment = (field.u * x + field.v * y) / norm
    return np.where(norm > 0, alignment, np.nan)


def divergence(field: VectorField) -> np.ndarray:
    """Divergence of the mean field, du/dx + dv/dy, by finite differences.

    Parameters
    ----------
    field : VectorField
        The binned field.

    Returns
    -------
    np.ndarray
        Divergence per bin, indexed ``[x_bin, y_bin]``. NaN next to bins
        with NaN values.
    """
    du_dx = np.gradient(field.u, field.x_centres, axis=0)
    dv_dy = np.gradient(field.v, field.y_centres, axis=1)
    return du_dx + dv_dy


def bin_density(
    focal: pd.DataFrame,
    x: str = "rel_x",
    y: str = "rel_y",
    bins: int | tuple[int, int] = 50,
    extent: Extent = None,
    normalise: bool = True,
) -> ScalarField:
    """Where neighbours are found relative to the focal.

    Parameters
    ----------
    focal : pd.DataFrame
        Focal-frame data, as returned by ``to_focal_frame``. Rows with NaN
        positions are ignored.
    x, y : str
        Columns holding the positions to bin.
    bins : int or tuple of int
        Number of bins along each axis, or ``(n_x, n_y)``.
    extent : float or tuple, optional
        Region to bin, as in ``bin_field``. Defaults to a symmetric square
        covering all positions.
    normalise : bool
        If True, return a probability density: counts divided by the number
        of samples inside the region and by the bin area, so it integrates
        to 1 over the region. If False, return raw counts.

    Returns
    -------
    ScalarField
        Density (or counts) per bin in ``values``, raw counts in ``counts``.

    Raises
    ------
    ValueError
        If either position column is missing.
    """
    missing = {x, y} - set(focal.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = focal[[x, y]].dropna()
    extent = _resolve_extent(data, extent)
    counts, x_edges, y_edges = np.histogram2d(data[x], data[y], bins=bins, range=extent)

    values = counts.copy()
    if normalise and counts.sum() > 0:
        bin_area = np.outer(np.diff(x_edges), np.diff(y_edges))
        values = counts / (counts.sum() * bin_area)

    return ScalarField(x_edges, y_edges, values, counts)
