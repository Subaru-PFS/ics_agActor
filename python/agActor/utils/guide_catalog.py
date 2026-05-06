"""Guide-catalog helpers: fetching, filtering, and position-tweaking.

This module owns the high-level ``get_guide_objects`` entry point and the
``filter_guide_objects`` / ``tweak_target_position`` utilities it delegates to.

Database queries are performed via :mod:`agActor.utils.queries`.
Dataclasses and constants live in :mod:`agActor.utils.data`.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pfs.utils.coordinates import updateTargetPosition
from pfs.utils.coordinates.CoordTransp import ag_pfimm_to_pixel
from pfs.utils.datamodel.ag import AutoGuiderStarMask

from agActor.utils.data import GuideCatalog
from agActor.utils.queries import (
    query_pfs_config_agc,
    query_pfs_design,
    query_pfs_design_agc,
)

logger = logging.getLogger(__name__)


def get_guide_objects(
    *,
    design_id: int,
    visit0: int | None = None,
    design_path: str | None = None,
    is_guide: bool = False,
    **kwargs,
) -> GuideCatalog:
    """Get the guide objects for a given design/visit.

    Parameters
    ----------
    design_id : int
        The PFS design ID.
    visit0 : int | None
        The PFS visit ID used to look up ``pfs_config_agc`` data.  When
        ``None`` or when no matching row exists, the latest ``pfs_design_agc``
        entry is used with position adjustments.
    design_path : str | None
        Path to the PFS design file.  Reserved for future use; currently
        ignored (guide objects are always fetched from opDB).
    is_guide : bool
        When ``True``, only high-quality GAIA stars are kept.
    **kwargs
        Optional field-centre / telescope-status overrides:

        - ``ra``, ``dec``, ``inst_pa`` – field centre coordinates and PA
        - ``taken_at``, ``inr``, ``adc``, ``m2_pos3`` – telescope status
        - ``dra``, ``ddec``, ``dinr`` – coordinate deltas (arcsec)

    Returns
    -------
    GuideCatalog
        Populated guide-catalog dataclass.
    """

    # Extract telescope status from kwargs
    taken_at = kwargs.get("taken_at")
    inr = kwargs.get("inr")
    adc = kwargs.get("adc")
    m2_pos3 = kwargs.get("m2_pos3", 6.0)
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")
    logger.info(f"taken_at={taken_at},inr={inr},adc={adc},m2_pos3={m2_pos3}")
    logger.info(f"design_id={design_id},design_path={design_path}")

    # Apply coordinate adjustments if provided
    if "dra" in kwargs and ra is not None:
        ra += kwargs.get("dra") / 3600
    if "ddec" in kwargs and dec is not None:
        dec += kwargs.get("ddec") / 3600
    if "dinr" in kwargs and inr is not None:
        inr += kwargs.get("dinr") / 3600

    logger.info(f"Getting guide_objects from pfs_config_agc via {design_id=} {visit0=}")
    field_design = query_pfs_design(design_id)

    # Fall back to field design values only if the caller did not supply them.
    # Use explicit `is None` checks rather than `or` to preserve a valid 0.0
    # (REFACTORING.md Issue 7).
    if ra is None:
        ra = field_design.field_ra
    if dec is None:
        dec = field_design.field_dec
    if inst_pa is None:
        inst_pa = field_design.field_inst_pa

    guide_objects = query_pfs_config_agc(
        design_id=design_id, visit0=visit0, as_dataframe=True
    )

    if len(guide_objects) == 0:
        logger.info(
            f"No pfs_config_agc entry for {visit0=}, using latest for {design_id=}"
        )
        guide_objects = query_pfs_design_agc(pfs_design_id=design_id, as_dataframe=True)

        if len(guide_objects) == 0:
            raise RuntimeError(f"No guide objects found for design_id={design_id}")

    # Apply telescope coordinate adjustments.
    guide_objects = tweak_target_position(
        guide_objects, ra, dec, inst_pa, taken_at or "now"
    )

    # Mark which guide objects should be filtered (only GALAXIES for now).
    logger.info(f"Guide objects before filtering: {len(guide_objects)}")
    guide_objects = filter_guide_objects(guide_objects, is_guide=is_guide)
    logger.info(
        f"Guide objects after filtering: {len(guide_objects.query('filtered_by == 0'))}"
    )

    return GuideCatalog(
        guide_objects=guide_objects,
        ra=ra,
        dec=dec,
        inr=inr,
        inst_pa=inst_pa,
        m2_pos3=m2_pos3,
        adc=adc,
        taken_at=taken_at,
    )


def tweak_target_position(
    guide_objects: pd.DataFrame,
    field_ra: float,
    field_dec: float,
    field_pa: float,
    obstime: datetime | str,
) -> pd.DataFrame:
    """Update RA/Dec and focal-plane positions for pfsDesign guide objects.

    Adjusts guide star positions for proper motion and parallax based on the
    observation time and field centre coordinates.

    See ``pfs.utils.pfsConfigUtils.tweakTargetPosition`` for reference.

    Parameters
    ----------
    guide_objects : pd.DataFrame
        DataFrame with columns ``ra``, ``dec``, ``pm_ra``, ``pm_dec``,
        ``parallax``, and ``agc_camera_id``.
    field_ra : float
        Right ascension of the field centre in degrees.
    field_dec : float
        Declination of the field centre in degrees.
    field_pa : float
        Position angle of the field in degrees.
    obstime : datetime | str
        Timezone-aware observation time, or the string ``"now"``.

    Returns
    -------
    pd.DataFrame
        Copy of the input with updated ``ra``, ``dec``, ``x``, and ``y``
        columns.

    Raises
    ------
    ValueError
        If *obstime* is a naive ``datetime`` (no timezone info).
    """

    guide_objects = guide_objects.copy()

    logger.info(
        f"Updating guide object positions from pfsDesign for telescope pointing "
        f"({field_ra=},{field_dec=},{field_pa=}) at {obstime=})"
    )

    cent = np.vstack([field_ra, field_dec])

    if isinstance(obstime, float):
        obstime = datetime.fromtimestamp(obstime, tz=timezone.utc)

    obstime = datetime.now(timezone.utc) if obstime == "now" else obstime

    if obstime.tzinfo is None or obstime.tzinfo.utcoffset(obstime) is None:
        raise ValueError("obstime must be timezone-aware (localized) or 'now'")

    # converting to ISO-8601
    obstime = obstime.isoformat(timespec='milliseconds').replace("+00:00", "Z")
    logger.info(f"obstime converted to ISO-8601 UTC: {obstime=}")

    # Updating ra/dec/position for guideStars objects.
    radec = np.vstack([guide_objects.ra, guide_objects.dec])
    pm = np.vstack([guide_objects.pm_ra, guide_objects.pm_dec])
    par = guide_objects.parallax.values

    guide_ra_now, guide_dec_now, guide_x_now, guide_y_now = (
        updateTargetPosition.update_target_position(
            radec, field_pa, cent, pm, par, obstime, mode="sky_pfi_ag"
        )
    )

    # converting to ag pixels
    guide_xy_pix = np.array(
        [
            ag_pfimm_to_pixel(agId, x, y)
            for agId, x, y in zip(guide_objects.agc_camera_id, guide_x_now, guide_y_now)
        ]
    )
    guide_x_pix = guide_xy_pix[:, 0].astype("float32")
    guide_y_pix = guide_xy_pix[:, 1].astype("float32")

    guide_objects["ra"] = guide_ra_now
    guide_objects["dec"] = guide_dec_now
    guide_objects["x"] = guide_x_pix
    guide_objects["y"] = guide_y_pix

    return guide_objects


def filter_guide_objects(
    guide_objects, is_guide=False, flag_column="flags"
) -> pd.DataFrame:
    """Apply filtering to guide objects based on their catalog flags.

    Always filters galaxies.  When *is_guide* is ``False`` (acquisition mode),
    also requires the full set of high-quality GAIA quality flags.

    Parameters
    ----------
    guide_objects : pd.DataFrame
        DataFrame containing guide object data including a column for flags.
    is_guide : bool, optional
        If ``True``, only galaxy filtering is applied (guiding mode).
        If ``False``, strict GAIA quality flags are also required (acquisition
        mode).  Default ``False``.
    flag_column : str, optional
        Name of the column that holds the integer flag bitmask.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an added ``filtered_by`` column (0 = keep).
    """
    guide_objects_df = guide_objects.copy()
    guide_objects_df["filtered_by"] = 0

    # Filter out the galaxies.
    try:
        galaxy_idx = (
            guide_objects_df[flag_column].values & AutoGuiderStarMask.GALAXY
        ) != 0
        guide_objects_df.loc[galaxy_idx, "filtered_by"] = (
            AutoGuiderStarMask.GALAXY.value
        )
        logger.info(f"Filtered {galaxy_idx.sum()} galaxies from results.")

        if is_guide is False:
            filters_for_inclusion = [
                AutoGuiderStarMask.NON_BINARY,
                AutoGuiderStarMask.GAIA,
                AutoGuiderStarMask.PHOTO_SIG,
                AutoGuiderStarMask.ASTROMETRIC,
                AutoGuiderStarMask.PMRA,
                AutoGuiderStarMask.PMDEC,
                AutoGuiderStarMask.PARA,
            ]

            # Mark stars that do NOT meet each inclusion requirement.
            for f in filters_for_inclusion:
                include_filter = (guide_objects_df[flag_column].values & f) == 0
                guide_objects_df.loc[include_filter, "filtered_by"] |= f.value
                logger.info(
                    f"Filtering non {f.name}, removes {include_filter.sum()} guide objects."
                )
    except KeyError:
        logger.warning(
            f"'flags' column not found in guide objects, "
            f"no filtering applied for {flag_column}."
        )
        guide_objects_df["flags"] = 0

    return guide_objects_df
