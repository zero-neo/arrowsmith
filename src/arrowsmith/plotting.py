"""Plot vector fields, profiles and direction distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from matplotlib.ticker import MaxNLocator

from .fields import ScalarField, VectorField, radial_alignment

FIGURE_DPI = 150
SAVE_DPI = 300

LABELS = {
    "rel_x": "Relative x",
    "rel_y": "Relative y",
    "distance": "Distance",
    "neighbour_vx": "Neighbour x velocity",
    "neighbour_vy": "Neighbour y velocity",
    "focal_ax": "Focal x acceleration",
    "focal_ay": "Focal y acceleration",
    "neighbour_ax": "Neighbour x acceleration",
    "neighbour_ay": "Neighbour y acceleration",
    "density": "Density",
    "focal_speed": "Focal speed",
    "radial_alignment": "Radial alignment",
    "magnitude": "Mean vector length",
}


def label_for(name: str) -> str:
    """Readable axis label for a column name.

    Parameters
    ----------
    name : str
        Column or quantity name, e.g. ``"rel_x"``.

    Returns
    -------
    str
        The label from ``LABELS``, or the name with underscores replaced by
        spaces and the first letter capitalised.
    """
    return LABELS.get(name, name.replace("_", " ").capitalize())


def _set_three_ticks(ax: Axes, round_limits: bool = False) -> None:
    """Put ticks only at both ends and the middle of each axis.

    With ``round_limits``, each axis is first widened to the nearest round
    numbers (e.g. 0.134 becomes 0.15), so the end ticks are readable.
    """
    for get_lim, set_lim, set_ticks in (
        (ax.get_xlim, ax.set_xlim, ax.set_xticks),
        (ax.get_ylim, ax.set_ylim, ax.set_yticks),
    ):
        lo, hi = get_lim()
        if round_limits:
            nice = MaxNLocator(nbins=4).tick_values(lo, hi)
            lo, hi = nice[0], nice[-1]
            set_lim(lo, hi)
        set_ticks([lo, (lo + hi) / 2, hi])
    # Use a true minus sign, as matplotlib does elsewhere.
    formatter = plt.FuncFormatter(
        lambda value, _: f"{value:.3g}".replace("-", "\u2212")
    )
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def _common_label(names: list[str]) -> str:
    """Shared y label for several quantities, e.g. "Focal acceleration"."""
    groups = {
        ("neighbour_vx", "neighbour_vy"): "Neighbour velocity",
        ("focal_ax", "focal_ay"): "Focal acceleration",
        ("neighbour_ax", "neighbour_ay"): "Neighbour acceleration",
    }
    for members, label in groups.items():
        if set(names) <= set(members):
            return label
    return "Mean"


def _get_ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=FIGURE_DPI, layout="constrained")
    return ax


def save_figure(
    figure: Figure | Axes,
    path: str | Path,
    dpi: int = SAVE_DPI,
    **savefig_kwargs,
) -> Path:
    """Save a figure at high resolution with tight margins.

    Parameters
    ----------
    figure : Figure or Axes
        The figure to save, or any axes on it.
    path : str or Path
        Output file. The extension sets the format: ``.png`` for images,
        ``.pdf`` or ``.svg`` for vector graphics, which stay sharp at any
        size.
    dpi : int
        Resolution for raster formats such as PNG, in dots per inch.
    **savefig_kwargs
        Passed to ``Figure.savefig``.

    Returns
    -------
    Path
        The path written to.
    """
    if isinstance(figure, Axes):
        figure = figure.figure
    path = Path(path)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    figure.savefig(path, dpi=dpi, **savefig_kwargs)
    return path


def _style_focal_axes(ax: Axes) -> None:
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(label_for("rel_x"))
    ax.set_ylabel(label_for("rel_y"))
    _set_three_ticks(ax)


def plot_quiver(
    field: VectorField,
    ax: Axes | None = None,
    color_by: str | np.ndarray | None = "radial_alignment",
    alpha_by_counts: bool = True,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    arrow_length: float | None = None,
    arrow_width: float | None = None,
    **quiver_kwargs,
) -> Axes:
    """Draw a vector field as arrows, optionally coloured and faded.

    Parameters
    ----------
    field : VectorField
        The binned field, as returned by ``bin_field``.
    ax : Axes, optional
        Axes to draw on. A new figure is created if omitted.
    color_by : str, np.ndarray or None
        ``"radial_alignment"``, ``"magnitude"``, an array indexed
        ``[x_bin, y_bin]`` like ``field.u``, or None for plain arrows.
    alpha_by_counts : bool
        Fade arrows in proportion to the number of samples in their bin,
        relative to the 95th percentile of non-empty bins.
    cmap : str
        Colour map used when ``color_by`` is set.
    vmin, vmax : float, optional
        Colour limits. Default to [-1, 1] for radial alignment, otherwise
        to the data range.
    colorbar : bool
        Add a colour bar when ``color_by`` is set.
    arrow_length : float, optional
        Arrow length per unit of vector magnitude, in plot (position)
        units: with ``arrow_length=0.1`` a vector of magnitude 1 is drawn
        0.1 long. Larger values give longer arrows. Defaults to
        matplotlib's automatic scaling.
    arrow_width : float, optional
        Shaft width as a fraction of the plot width (matplotlib's
        ``width``). Defaults to matplotlib's automatic width.
    **quiver_kwargs
        Passed to ``Axes.quiver``, e.g. ``scale`` or ``width``.

    Returns
    -------
    Axes
        The axes drawn on. Ticks sit only at both ends and the middle of
        each axis, and labels use readable names (see ``LABELS``).
    """
    ax = _get_ax(ax)
    x, y = field.mesh()

    if isinstance(color_by, str):
        if color_by == "radial_alignment":
            colors = radial_alignment(field)
            vmin = -1.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        elif color_by == "magnitude":
            colors = field.magnitude
        else:
            raise ValueError(
                "color_by must be 'radial_alignment', 'magnitude', an array "
                f"or None, not {color_by!r}"
            )
    else:
        colors = color_by

    args = [x.ravel(), y.ravel(), field.u.ravel(), field.v.ravel()]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != field.u.shape:
            raise ValueError(
                f"color_by array has shape {colors.shape}, expected {field.u.shape}"
            )
        args.append(colors.ravel())
        quiver_kwargs.update(cmap=cmap, clim=(vmin, vmax))

    if alpha_by_counts and field.counts.max() > 0:
        # Scale to the 95th percentile so a few crowded bins near the origin
        # don't make every other arrow nearly transparent.
        reference = np.percentile(field.counts[field.counts > 0], 95)
        alpha = np.clip(field.counts / reference, 0, 1)
        quiver_kwargs.setdefault("alpha", alpha.ravel())
    quiver_kwargs.setdefault("angles", "xy")
    if arrow_length is not None:
        quiver_kwargs.update(scale_units="xy", scale=1 / arrow_length)
    if arrow_width is not None:
        quiver_kwargs["width"] = arrow_width

    quiver = ax.quiver(*[np.ma.masked_invalid(a) for a in args], **quiver_kwargs)
    if colors is not None and colorbar:
        label = label_for(color_by) if isinstance(color_by, str) else None
        cbar = ax.figure.colorbar(quiver, ax=ax, label=label)
        lo, hi = cbar.mappable.get_clim()
        cbar.set_ticks([lo, (lo + hi) / 2, hi])

    ax.set_xlim(field.x_edges[0], field.x_edges[-1])
    ax.set_ylim(field.y_edges[0], field.y_edges[-1])
    _style_focal_axes(ax)
    return ax


def plot_profile(
    focal: pd.DataFrame,
    quantity: str | list[str] = "neighbour_vx",
    along: str = "rel_x",
    bins: int = 30,
    value_range: tuple[float, float] | None = None,
    min_count: int = 10,
    kind: str = "mean",
    ax: Axes | None = None,
    **plot_kwargs,
) -> Axes:
    """Plot a quantity against one position coordinate.

    Parameters
    ----------
    focal : pd.DataFrame
        Focal-frame data, as returned by ``to_focal_frame``.
    quantity : str or list of str
        Column(s) to plot on the y axis. Defaults to the neighbour's forward
        velocity; use ``"neighbour_vy"`` for sideways velocity, or
        ``["neighbour_vx", "neighbour_vy"]`` for both as separate lines
        with a legend (``kind="mean"`` only).
    along : str
        Column for the x axis, or ``"distance"`` for the focal-neighbour
        distance computed from ``rel_x`` and ``rel_y``.
    bins : int
        Number of bins along ``along`` (or hexbin grid size).
    value_range : tuple of float, optional
        Range of ``along`` to include. Defaults to the data range.
    min_count : int
        Bins with fewer samples are left out of the mean profile.
    kind : str
        ``"mean"`` for the binned mean with a band of one standard error,
        or ``"hexbin"`` for the density of all samples.
    ax : Axes, optional
        Axes to draw on. A new figure is created if omitted.
    **plot_kwargs
        Passed to ``Axes.plot`` (mean) or ``Axes.hexbin``.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    quantities = [quantity] if isinstance(quantity, str) else list(quantity)
    if kind == "hexbin" and len(quantities) > 1:
        raise ValueError("kind='hexbin' takes a single quantity")

    position = (
        np.hypot(focal["rel_x"], focal["rel_y"])
        if along == "distance"
        else focal[along]
    )
    if value_range is None:
        lo, hi = np.nanmin(position), np.nanmax(position)
    else:
        lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1)

    for name in quantities:
        data = pd.DataFrame({"position": position, "value": focal[name]}).dropna()
        data = data[(data["position"] >= lo) & (data["position"] <= hi)]

        if kind == "hexbin":
            plot_kwargs.setdefault("mincnt", 1)
            ax.hexbin(data["position"], data["value"], gridsize=bins, **plot_kwargs)
        elif kind == "mean":
            groups = data.groupby(
                pd.cut(data["position"], edges, include_lowest=True), observed=True
            )["value"]
            stats = groups.agg(["mean", "std", "count"])
            stats = stats[stats["count"] >= min_count]
            centres = np.array([interval.mid for interval in stats.index])
            sem = stats["std"] / np.sqrt(stats["count"])
            (line,) = ax.plot(
                centres, stats["mean"], label=label_for(name), **plot_kwargs
            )
            ax.fill_between(
                centres,
                stats["mean"] - sem,
                stats["mean"] + sem,
                color=line.get_color(),
                alpha=0.3,
                linewidth=0,
            )
        else:
            raise ValueError(f"kind must be 'mean' or 'hexbin', not {kind!r}")

    if len(quantities) == 1:
        ax.set_ylabel(label_for(quantities[0]))
    else:
        ax.legend()
        ax.set_ylabel(_common_label(quantities))
    ax.set_xlabel(label_for(along))
    ax.set_xlim(lo, hi)
    _set_three_ticks(ax, round_limits=True)
    return ax


def angle_histograms(
    focal: pd.DataFrame,
    u: str = "neighbour_vx",
    v: str = "neighbour_vy",
    x: str = "rel_x",
    y: str = "rel_y",
    bins: int = 6,
    extent: float = 1.0,
    angle_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count vector directions within each cell of a coarse spatial grid.

    Parameters
    ----------
    focal : pd.DataFrame
        Focal-frame data, as returned by ``to_focal_frame``.
    u, v : str
        Columns holding the vector components.
    x, y : str
        Columns holding the positions.
    bins : int
        Number of spatial cells along each axis.
    extent : float
        Half-width of the square region, ``[-extent, extent]``.
    angle_bins : int
        Number of direction bins over the full circle.

    Returns
    -------
    counts : np.ndarray
        Shape ``(bins, bins, angle_bins)``, indexed ``[x_bin, y_bin, angle]``.
    centres : np.ndarray
        Spatial cell centres (same along x and y).
    angle_edges : np.ndarray
        Direction bin edges in radians, from -pi to pi.
    """
    data = focal[[x, y, u, v]].dropna()
    edges = np.linspace(-extent, extent, bins + 1)
    angle_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)

    ix = np.digitize(data[x], edges) - 1
    iy = np.digitize(data[y], edges) - 1
    angles = np.arctan2(data[v], data[u])
    ia = np.clip(np.digitize(angles, angle_edges) - 1, 0, angle_bins - 1)

    inside = (ix >= 0) & (ix < bins) & (iy >= 0) & (iy < bins)
    counts = np.zeros((bins, bins, angle_bins))
    np.add.at(counts, (ix[inside], iy[inside], ia[inside]), 1)

    centres = (edges[:-1] + edges[1:]) / 2
    return counts, centres, angle_edges


def plot_polar_grid(
    focal: pd.DataFrame,
    u: str = "neighbour_vx",
    v: str = "neighbour_vy",
    bins: int = 6,
    extent: float = 1.0,
    angle_bins: int = 12,
    ax: Axes | None = None,
    color: str = "C0",
    **patch_kwargs,
) -> Axes:
    """Draw a rose diagram of vector directions in each cell of a grid.

    Each rose shows the full distribution of directions in its cell, not
    just the mean. Roses are normalised per cell, so they show the shape of
    the distribution rather than how many samples each cell holds.

    Parameters
    ----------
    focal : pd.DataFrame
        Focal-frame data, as returned by ``to_focal_frame``.
    u, v : str
        Columns holding the vector components.
    bins : int
        Number of cells along each axis. Keep this small (about 4-10).
    extent : float
        Half-width of the square region, ``[-extent, extent]``.
    angle_bins : int
        Number of direction bins (petals) per rose.
    ax : Axes, optional
        Axes to draw on. A new figure is created if omitted.
    color : str
        Petal colour.
    **patch_kwargs
        Passed to ``PatchCollection``, e.g. ``edgecolor``.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    counts, centres, angle_edges = angle_histograms(
        focal, u=u, v=v, bins=bins, extent=extent, angle_bins=angle_bins
    )
    max_radius = 0.45 * (centres[1] - centres[0]) if bins > 1 else 0.45 * extent
    theta = np.degrees(angle_edges)

    wedges = []
    for i, cx in enumerate(centres):
        for j, cy in enumerate(centres):
            peak = counts[i, j].max()
            if peak == 0:
                continue
            for k, n in enumerate(counts[i, j]):
                if n > 0:
                    radius = max_radius * n / peak
                    wedges.append(Wedge((cx, cy), radius, theta[k], theta[k + 1]))

    patch_kwargs.setdefault("edgecolor", "none")
    ax.add_collection(PatchCollection(wedges, facecolor=color, **patch_kwargs))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    _style_focal_axes(ax)
    return ax


def plot_density(
    field: ScalarField,
    ax: Axes | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    label: str = "density",
    **pcolormesh_kwargs,
) -> Axes:
    """Draw a scalar field, such as neighbour density, as a heat map.

    Parameters
    ----------
    field : ScalarField
        The binned field, as returned by ``bin_density``.
    ax : Axes, optional
        Axes to draw on. A new figure is created if omitted.
    cmap : str
        Colour map.
    colorbar : bool
        Add a colour bar.
    label : str
        Colour bar label, looked up in ``LABELS``.
    **pcolormesh_kwargs
        Passed to ``Axes.pcolormesh``, e.g. ``vmin``, ``vmax`` or
        ``norm=matplotlib.colors.LogNorm()``.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    ax = _get_ax(ax)
    if "norm" not in pcolormesh_kwargs:
        pcolormesh_kwargs.setdefault("vmin", 0)
    # pcolormesh expects values indexed [y, x], so transpose here only.
    mesh = ax.pcolormesh(
        field.x_edges, field.y_edges, field.values.T, cmap=cmap, **pcolormesh_kwargs
    )
    if colorbar:
        cbar = ax.figure.colorbar(mesh, ax=ax, label=label_for(label))
        lo, hi = mesh.get_clim()
        cbar.set_ticks([lo, (lo + hi) / 2, hi])
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.3g}")
        )

    ax.set_xlim(field.x_edges[0], field.x_edges[-1])
    ax.set_ylim(field.y_edges[0], field.y_edges[-1])
    _style_focal_axes(ax)
    return ax
