"""The sky transparency the guide stars of an AG exposure show, for the log.

Nothing here feeds acquisition or guiding. It reads what it needs with its own two queries
(the spots' magnitudes, the guide stars' colours), and `log_transparency` turns any failure
into a log line.

The number is ``pfs.utils.agMetrics``': the stars' measured over expected flux, the median
over the brighter stars of the exposure. It is not calibrated to read 1 under a clear sky, so
the log gives it twice: as it is, and relative to the first exposures of the same visit, which
is what shows cloud coming in over a field.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
from agActor.utils.data import query_db
from pfs.utils import agMetrics

logger = logging.getLogger(__name__)

#: The first exposures of a visit that set what its later ones are compared with.
N_REFERENCE_EXPOSURES = 5


def query_spot_magnitudes(frame_id: int) -> pd.DataFrame:
    """Return ``agc_camera_id``, ``spot_id`` and ``estimated_magnitude`` of an AG exposure's spots."""
    sql = """
SELECT agc_camera_id, spot_id, estimated_magnitude
FROM agc_data
WHERE agc_exposure_id = :agc_exposure_id
"""
    return query_db(sql, {"agc_exposure_id": int(frame_id)})


@functools.lru_cache(maxsize=4)
def query_guide_star_colors(design_id: int) -> pd.Series:
    """Return the Gaia BP-RP of a design's guide stars, indexed by ``guide_star_id``.

    Read once per design. Empty for a design that is not in ``pfs_design_agc``.
    """
    sql = """
SELECT guide_star_id, guide_star_color
FROM pfs_design_agc
WHERE pfs_design_id = :pfs_design_id
"""
    colors = query_db(sql, {"pfs_design_id": int(design_id)})
    return (
        colors.drop_duplicates("guide_star_id").set_index("guide_star_id")["guide_star_color"].astype(float)
    )


def frame_transparency(
    detected_objects: pd.DataFrame,
    identified_objects: pd.DataFrame,
    guide_objects: pd.DataFrame,
    spot_magnitudes: pd.DataFrame,
    colors: pd.Series,
) -> agMetrics.AgTransparency:
    """Return the transparency of one AG exposure from its detections matched to guide stars.

    Parameters
    ----------
    detected_objects : pd.DataFrame
        The exposure's detections, with ``agc_camera_id`` and ``spot_id``.
    identified_objects : pd.DataFrame
        The astrometric fit's output: ``detected_object_id`` and ``guide_object_id`` (the labels
        of the rows in ``detected_objects`` and ``guide_objects``) and ``matched``. Only the
        rows with ``matched == 1`` are used.
    guide_objects : pd.DataFrame
        The guide stars, with ``source_id`` and ``mag``, the catalogue magnitude.
    spot_magnitudes : pd.DataFrame
        ``agc_camera_id``, ``spot_id`` and ``estimated_magnitude`` (`query_spot_magnitudes`).
    colors : pd.Series
        The guide stars' Gaia BP-RP by ``source_id`` (`query_guide_star_colors`).

    Returns
    -------
    pfs.utils.agMetrics.AgTransparency
        The median transparency and the number of stars behind it; NaN and 0 when no matched
        star has a magnitude and a colour.
    """
    matched = identified_objects[identified_objects["matched"] == 1]
    if len(matched) == 0:
        return agMetrics.AgTransparency(transparency=float("nan"), nStars=0)
    spots = detected_objects.loc[matched["detected_object_id"].to_numpy(), ["agc_camera_id", "spot_id"]]
    stars = guide_objects.loc[matched["guide_object_id"].to_numpy(), ["source_id", "mag"]]
    magnitudes = spot_magnitudes.drop_duplicates(["agc_camera_id", "spot_id"]).set_index(
        ["agc_camera_id", "spot_id"]
    )
    keys = pd.MultiIndex.from_arrays([spots["agc_camera_id"].to_numpy(), spots["spot_id"].to_numpy()])
    estimated = magnitudes["estimated_magnitude"].reindex(keys).to_numpy(dtype=float)
    color = colors.reindex(stars["source_id"].to_numpy()).to_numpy(dtype=float)
    transparency = agMetrics.magnitudesToTransparency(estimated, stars["mag"].to_numpy(dtype=float), color)
    return agMetrics.summarizeTransparency(transparency, estimated)


class TransparencyReference:
    """What a visit's exposures are compared with: the median of its first few transparencies.

    Parameters
    ----------
    n_first : int
        How many finite values of a visit make its reference.
    """

    def __init__(self, n_first: int = N_REFERENCE_EXPOSURES):
        self.n_first = n_first
        self._key = None
        self._first: list[float] = []

    def relative(self, key, transparency: float) -> float:
        """Return ``transparency`` over the reference of ``key``, a visit; NaN until there is one.

        A new ``key`` starts a new reference. The exposures that make the reference get a
        value too, relative to the median of those so far, so the first reads 1.
        """
        if key != self._key:
            self._key, self._first = key, []
        if not np.isfinite(transparency):
            return float("nan")
        if len(self._first) < self.n_first:
            self._first.append(float(transparency))
        return float(transparency / np.median(self._first))

    @property
    def n_used(self) -> int:
        """How many exposures the present reference is made of."""
        return len(self._first)


def describe_transparency(
    frame_id: int, transparency: agMetrics.AgTransparency, relative: float | None = None, n_reference: int = 0
) -> str:
    """Return the log line for one exposure's transparency."""
    if transparency.nStars == 0:
        return (
            f"Transparency: frame_id={frame_id} not measured (no matched star with a magnitude and a colour)"
        )
    line = f"Transparency: frame_id={frame_id} transparency={transparency.transparency:.3f} (n={transparency.nStars}, uncalibrated)"
    if relative is not None and np.isfinite(relative):
        line += f" relative={relative:.3f} (to the visit's first {n_reference})"
    return line


def log_transparency(
    log: logging.Logger,
    guide_offsets,
    *,
    frame_id: int,
    design_id: int | None,
    visit_id: int | None = None,
    reference: TransparencyReference | None = None,
) -> None:
    """Log the transparency of the exposure behind ``guide_offsets``. Never raises.

    Parameters
    ----------
    log : logging.Logger
        Where the line goes.
    guide_offsets : agActor.utils.data.GuideOffsets
        The exposure's detections, matches and guide stars.
    frame_id : int
        The AG exposure.
    design_id : int | None
        The design the guide stars' colours are read for; without it nothing is measured.
    visit_id : int | None
        The visit, which with the design says when ``reference`` starts again.
    reference : TransparencyReference | None
        Given by the guide loop to add the value relative to the visit's first exposures.
    """
    try:
        if design_id is None:
            log.info(f"Transparency: frame_id={frame_id} not measured (no design id)")
            return
        transparency = frame_transparency(
            guide_offsets.detected_objects,
            guide_offsets.identified_objects,
            guide_offsets.guide_objects,
            query_spot_magnitudes(frame_id),
            query_guide_star_colors(int(design_id)),
        )
        relative, n_reference = None, 0
        if reference is not None:
            relative = reference.relative((design_id, visit_id), transparency.transparency)
            n_reference = reference.n_used
        log.info(describe_transparency(frame_id, transparency, relative, n_reference))
    except Exception as e:  # noqa: BLE001  whatever fails here must not reach the guide loop
        log.warning(f"Transparency: frame_id={frame_id} not measured: {e!r}")
