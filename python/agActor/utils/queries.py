"""Database query and write helpers for the AG actor.

All ``query_*`` and ``write_*`` functions that touch opDB (or the Gaia DB)
live here, together with the thin helpers that retrieve telescope status or
detected objects via those queries.

The dataclasses (``GuideCatalog``, ``GuideOffsets``) and constants
(``BAD_DETECTION_FLAGS``) remain in :mod:`agActor.utils.data`.
"""

import logging
from datetime import datetime
from enum import IntFlag

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from numpy.typing import NDArray

from pfs.utils.database.db import DB
from pfs.utils.database.opdb import OpDB
from pfs.utils.database.gaia import GaiaDB
from pfs.utils.datamodel.ag import SourceDetectionFlag

logger = logging.getLogger(__name__)

# Defined here (not in data.py) so that queries.py has no imports from data.py,
# breaking the potential circular dependency.  data.py re-exports both symbols.

BAD_DETECTION_FLAGS = (
    SourceDetectionFlag.SATURATED
    | SourceDetectionFlag.EDGE
    | SourceDetectionFlag.BAD_ELLIP
    | SourceDetectionFlag.FLAT_TOP
)


class GuideOffsetFlag(IntFlag):
    """Flags stored in the ``mask`` column of ``agc_guide_offset``.

    Attributes
    ----------
    OK : int
        Guide offset is valid.
    INVALID_OFFSET : int
        Guide offset was computed but not applied.
    """

    OK = 0x0000
    INVALID_OFFSET = 0x0001


# ---------------------------------------------------------------------------
# Generic DB helper
# ---------------------------------------------------------------------------


def query_db(
    sql: str,
    params: dict | list | None = None,
    as_dataframe: bool = True,
    single_as_series: bool = False,
    db: DB | None = None,
) -> pd.DataFrame | pd.Series | np.ndarray | None:
    """Execute a SQL query and return the results.

    Parameters
    ----------
    sql : str
        The SQL query to execute.
    params : dict | list | None
        Bind parameters for the query.
    as_dataframe : bool
        When ``True`` return a :class:`pandas.DataFrame` (default).  If a
        single row is returned and *single_as_series* is also ``True``, return
        a :class:`pandas.Series` instead.
    single_as_series : bool
        Return a :class:`pandas.Series` when exactly one row is returned.
        Ignored when *as_dataframe* is ``False``.
    db : DB | None
        Database connection to use.  Defaults to :class:`~pfs.utils.database.opdb.OpDB`.

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray | None
        Results of the query.
    """
    db = db or OpDB()
    if as_dataframe:
        result = db.query_dataframe(sql, params=params)
        if len(result) == 1 and single_as_series:
            result = result.iloc[0]
    else:
        result = db.query_array(sql, params=params)
        if len(result) == 1:
            result = result[0]

    return result


# ---------------------------------------------------------------------------
# Table-specific queries
# ---------------------------------------------------------------------------


def query_agc_data(agc_exposure_id: int, as_dataframe: bool = True, **kwargs):
    """Query ``agc_data`` for a given exposure.

    Parameters
    ----------
    agc_exposure_id : int
        The AGC exposure ID.
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | np.ndarray
    """
    sql = """
SELECT agc_camera_id,
 spot_id,
 image_moment_00_pix,
 centroid_x_pix,
 centroid_y_pix,
 central_image_moment_11_pix,
 central_image_moment_20_pix,
 central_image_moment_02_pix,
 peak_pixel_x_pix,
 peak_pixel_y_pix,
 peak_intensity,
 background,
 COALESCE(flags, CAST(centroid_x_pix >= 511.5 + 24 AS INTEGER)) AS flags
FROM agc_data
WHERE agc_exposure_id = :agc_exposure_id
ORDER BY agc_camera_id, spot_id
"""
    params = {"agc_exposure_id": agc_exposure_id}
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_tel_status(
    pfs_visit_id: int, status_sequence_id: int, as_dataframe: bool = True, **kwargs
):
    """Query ``tel_status`` for a given visit and sequence.

    Parameters
    ----------
    pfs_visit_id : int
        PFS visit ID.
    status_sequence_id : int
        Sequence ID within the visit.
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray
    """
    sql = """
SELECT
    altitude,
    azimuth,
    insrot,
    adc_pa,
    m2_pos3,
    tel_ra,
    tel_dec,
    dome_shutter_status,
    dome_light_status,
    created_at AT TIME ZONE 'Pacific/Honolulu' AS created_at
FROM 
    tel_status
WHERE 
    pfs_visit_id=:pfs_visit_id
  AND 
    status_sequence_id=:status_sequence_id
"""
    params = {"pfs_visit_id": pfs_visit_id, "status_sequence_id": status_sequence_id}
    return query_db(
        sql, params, as_dataframe=as_dataframe, single_as_series=True, **kwargs
    )


def query_agc_exposure(agc_exposure_id: int, as_dataframe: bool = True, **kwargs):
    """Query ``agc_exposure`` joined with ``pfs_visit`` for a given exposure.

    Parameters
    ----------
    agc_exposure_id : int
        The AGC exposure ID.
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray
    """
    sql = """
SELECT
    t0.agc_exposure_id,
    t0.pfs_visit_id,
    t1.pfs_design_id,
    t0.agc_exptime,
    t0.taken_at AT TIME ZONE 'Pacific/Honolulu' AS taken_at,
    t0.azimuth,
    t0.altitude,
    t0.insrot,
    t0.adc_pa,
    t0.outside_temperature,
    t0.outside_humidity,
    t0.outside_pressure,
    t0.m2_pos3
FROM 
    agc_exposure t0, pfs_visit t1
WHERE 
    t0.pfs_visit_id=t1.pfs_visit_id
    AND
    t0.agc_exposure_id=:agc_exposure_id
"""
    params = {"agc_exposure_id": agc_exposure_id}
    return query_db(
        sql, params, as_dataframe=as_dataframe, single_as_series=True, **kwargs
    )


def query_pfs_design_agc(pfs_design_id: int, as_dataframe: bool = True, **kwargs):
    """Query ``pfs_design_agc`` for a given design.

    Parameters
    ----------
    pfs_design_id : int
        PFS design ID.
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | np.ndarray
    """
    sql = """
SELECT
    guide_star_id as source_id,
    guide_star_ra as ra,
    guide_star_dec as dec,
    guide_star_pm_ra as pm_ra,
    guide_star_pm_dec as pm_dec,
    guide_star_parallax as parallax,
    guide_star_magnitude as mag,
    agc_camera_id as agc_camera_id,
    agc_target_x_pix as x,
    agc_target_y_pix as y,
    guide_star_flag as flags
FROM pfs_design_agc
WHERE pfs_design_id=:pfs_design_id
ORDER BY guide_star_id
"""
    params = {"pfs_design_id": pfs_design_id}
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_pfs_config_agc(
    *, design_id: int, visit0: int, as_dataframe: bool = True, **kwargs
):
    """Get the guide star configuration for a given PFS design and visit.

    Parameters
    ----------
    design_id : int
        The PFS design ID.
    visit0 : int
        The PFS visit ID (``visit0`` column in ``pfs_config_agc``).
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | np.ndarray
    """
    sql = """
SELECT 
    t0.guide_star_id as source_id,
    t1.guide_star_ra as ra,
    t1.guide_star_dec as dec,
    t1.guide_star_pm_ra as pm_ra,
    t1.guide_star_pm_dec as pm_dec,
    t1.guide_star_parallax as parallax,                
    t1.guide_star_magnitude as mag,
    t0.agc_camera_id as agc_camera_id,
    t0.agc_final_x_pix as x,
    t0.agc_final_y_pix as y,
    t1.guide_star_flag as flags
FROM pfs_config_agc t0, pfs_design_agc t1
WHERE t0.pfs_design_id = t1.pfs_design_id
    AND t0.guide_star_id = t1.guide_star_id
    AND t0.pfs_design_id = :pfs_design_id
    AND t0.visit0 = :visit0
ORDER BY t0.guide_star_id
          """
    params = {"pfs_design_id": design_id, "visit0": visit0}
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_pfs_design(pfs_design_id: int, as_dataframe: bool = True, **kwargs):
    """Query ``pfs_design`` and add convenient field-centre aliases.

    Parameters
    ----------
    pfs_design_id : int
        PFS design ID.
    as_dataframe : bool
        Return a :class:`~pandas.DataFrame` when ``True`` (default).

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray
    """
    sql = """
          SELECT *, ra_center_designed as field_ra,
                 dec_center_designed as field_dec,
                 pa_designed as field_inst_pa
          FROM pfs_design
          WHERE pfs_design_id = :pfs_design_id
          """
    params = {"pfs_design_id": pfs_design_id}
    return query_db(
        sql, params, as_dataframe=as_dataframe, single_as_series=True, **kwargs
    )


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_agc_guide_offset(
    *,
    frame_id: int,
    taken_at: datetime | float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    pa: float | None = None,
    delta_ra: float | None = None,
    delta_dec: float | None = None,
    delta_insrot: float | None = None,
    delta_scale: float | None = None,
    delta_az: float | None = None,
    delta_el: float | None = None,
    delta_z: float | None = None,
    delta_zs: NDArray | None = None,
    offset_flags: GuideOffsetFlag = GuideOffsetFlag.OK,
    db: DB | None = None,
):
    """Write the guide offsets to the database.

    If a value is not passed to the function then the default ``None`` will be
    written to the database.

    Parameters
    ----------
    frame_id : int
        The frame id of the frame.
    taken_at : datetime | float | None
        The time the frame was taken.  A Unix timestamp (float) is converted
        to a local naive datetime.
    ra : float | None
        Right ascension of the field in degrees.
    dec : float | None
        Declination of the field in degrees.
    pa : float | None
        Instrument position angle in degrees.
    delta_ra : float | None
        Right ascension offset in arcseconds.
    delta_dec : float | None
        Declination offset in arcseconds.
    delta_insrot : float | None
        Instrument rotator offset in arcseconds.
    delta_scale : float | None
        Scale change.
    delta_el : float | None
        Elevation (Altitude) offset in arcseconds.
    delta_az : float | None
        Azimuth offset in arcseconds.
    delta_z : float | None
        Average focus offset.
    delta_zs : NDArray | None
        Focus offset per camera.
    offset_flags : GuideOffsetFlag
        Any flags for the data, stored in the ``mask`` column.
        Defaults to :attr:`GuideOffsetFlag.OK`.
    db : DB | None
        The database to use.  Defaults to
        :class:`~pfs.utils.database.opdb.OpDB`.
    """
    db = db or OpDB()
    try:
        if isinstance(taken_at, (int, float)):
            taken_at = datetime.fromtimestamp(taken_at)
        params = dict(
            agc_exposure_id=frame_id,
            taken_at=taken_at if taken_at is not None else None,
            guide_ra=float(ra) if ra is not None else None,
            guide_dec=float(dec) if dec is not None else None,
            guide_pa=float(pa) if pa is not None else None,
            guide_delta_ra=float(delta_ra) if delta_ra is not None else None,
            guide_delta_dec=float(delta_dec) if delta_dec is not None else None,
            guide_delta_insrot=float(delta_insrot) if delta_insrot is not None else None,
            guide_delta_scale=float(delta_scale) if delta_scale is not None else None,
            guide_delta_az=float(delta_az) if delta_az is not None else None,
            guide_delta_el=float(delta_el) if delta_el is not None else None,
            mask=offset_flags.value,
            guide_delta_z=float(delta_z) if delta_z is not None else None,
        )
        if delta_zs is not None:
            params.update(guide_delta_z1=float(delta_zs[0]))
            params.update(guide_delta_z2=float(delta_zs[1]))
            params.update(guide_delta_z3=float(delta_zs[2]))
            params.update(guide_delta_z4=float(delta_zs[3]))
            params.update(guide_delta_z5=float(delta_zs[4]))
            params.update(guide_delta_z6=float(delta_zs[5]))

        logger.info(f"Writing agc_guide_offsets with {params=}")
        db.insert_kw("agc_guide_offset", **params)
    except Exception as e:
        logger.warning(f"Failed to write agc_guide_offsets: {e}")
        raise e


def write_agc_match(
    *,
    design_id: int,
    frame_id: int,
    guide_objects: pd.DataFrame,
    detected_objects: pd.DataFrame,
    identified_objects: pd.DataFrame,
    db: DB | None = None,
) -> int | None:
    """Insert AG identified objects into ``opdb.agc_match``.

    Parameters
    ----------
    design_id : int
        The PFS design ID.
    frame_id : int
        The exposure ID for the AGC frame.
    guide_objects : pd.DataFrame
        Guide star data.
    detected_objects : pd.DataFrame
        Detected object data.
    identified_objects : pd.DataFrame
        Matched guide/detected objects.
    db : DB | None
        The database to use.  Defaults to
        :class:`~pfs.utils.database.opdb.OpDB`.

    Returns
    -------
    int | None
        Number of rows inserted, or ``None`` if no matches.
    """
    db = db or OpDB()
    try:
        rows_to_insert = []
        for idx, match in identified_objects.iterrows():
            detected_idx = int(match.detected_object_id)
            guide_idx = int(match.guide_object_id)

            nominal_x_mm = float(match.guide_object_x_mm)
            center_x_mm = float(match.detected_object_x_mm)

            # TODO: move negative, see INSTRM-2654
            nominal_y_mm = float(match.guide_object_y_mm) * -1
            center_y_mm = float(match.detected_object_y_mm) * -1

            row = {
                "pfs_design_id": design_id,
                "agc_exposure_id": frame_id,
                "agc_camera_id": int(detected_objects["agc_camera_id"][detected_idx]),
                "spot_id": int(detected_objects["spot_id"][detected_idx]),
                "guide_star_id": int(guide_objects["source_id"][guide_idx]),
                "agc_nominal_x_mm": float(nominal_x_mm),
                "agc_nominal_y_mm": float(nominal_y_mm),
                "agc_center_x_mm": float(center_x_mm),
                "agc_center_y_mm": float(center_y_mm),
                "flags": int(match["matched"]),
            }
            rows_to_insert.append(row)

        if rows_to_insert:
            df = pd.DataFrame(rows_to_insert)
            logger.debug("Inserting data into database")
            n_rows = db.insert_dataframe(df=df, table="agc_match")
            logger.info(f"Finished inserting agc_match data: {n_rows} rows inserted")

            return n_rows
    except Exception as e:
        logger.warning(f"Failed to insert agc_match data: {e}")
        raise e

    return None


# ---------------------------------------------------------------------------
# Compound helpers that query the DB but are not simple table wrappers
# ---------------------------------------------------------------------------


def get_telescope_status(*, frame_id, **kwargs):
    """Get the telescope status information for a specific frame ID.

    Parameters
    ----------
    frame_id : int
        The frame ID to retrieve telescope status for.
    **kwargs
        Optional overrides:

        - ``taken_at`` – timestamp when the frame was taken
        - ``inr`` – instrument rotator angle
        - ``adc`` – atmospheric dispersion corrector value
        - ``m2_pos3`` – secondary mirror position
        - ``sequence_id`` – sequence ID for the ``tel_status`` table

    Returns
    -------
    tuple[datetime | None, float | None, float | None, float | None]
        ``(taken_at, inr, adc, m2_pos3)``

    Notes
    -----
    Caller-supplied values are *always* preferred over DB values.  Explicit
    ``is None`` guards are used so that a valid ``0.0`` is never silently
    replaced by the DB value (REFACTORING.md Issue 7).
    """
    logger.debug(f"Getting telescope status for {frame_id=}")

    # Extract values from kwargs if provided
    taken_at = kwargs.get("taken_at")
    inr = kwargs.get("inr")
    adc = kwargs.get("adc")
    m2_pos3 = kwargs.get("m2_pos3")

    # Check if we need to fetch any missing values from the database
    if any(value is None for value in (taken_at, inr, adc, m2_pos3)):
        # First, query the agc_exposure table to get basic information, including visit_id.
        logger.info(f"Getting agc_exposure from opdb for frame_id={frame_id}")
        agc_exposure_info = query_agc_exposure(frame_id, as_dataframe=True)
        visit_id = int(agc_exposure_info.pfs_visit_id) if pd.notna(agc_exposure_info.pfs_visit_id) else 0
        db_taken_at = agc_exposure_info.taken_at
        db_inr = float(agc_exposure_info.insrot) if pd.notna(agc_exposure_info.insrot) else None
        db_adc = float(agc_exposure_info.adc_pa) if pd.notna(agc_exposure_info.adc_pa) else None
        db_m2_pos3 = float(agc_exposure_info.m2_pos3) if pd.notna(agc_exposure_info.m2_pos3) else None

        # If sequence_id is provided, get more accurate information from tel_status table
        sequence_id = kwargs.get("sequence_id")
        if sequence_id is not None:
            logger.info(
                f"Getting telescope status from opdb for {visit_id=},{sequence_id=}"
            )
            tel_status_info = query_tel_status(visit_id, sequence_id, as_dataframe=True)
            db_inr = float(tel_status_info.insrot) if pd.notna(tel_status_info.insrot) else db_inr
            db_adc = float(tel_status_info.adc_pa) if pd.notna(tel_status_info.adc_pa) else db_adc
            db_m2_pos3 = float(tel_status_info.m2_pos3) if pd.notna(tel_status_info.m2_pos3) else db_m2_pos3
            db_taken_at = tel_status_info.created_at

        # Use database values for any missing parameters.
        # Use explicit `is None` checks rather than `or` so that valid numeric
        # zero values (e.g. inr=0.0, adc=0.0) supplied by the caller are never
        # silently overwritten by the DB value (REFACTORING.md Issue 7).
        if taken_at is None:
            taken_at = db_taken_at
        if inr is None:
            inr = db_inr
        if adc is None:
            adc = db_adc
        if m2_pos3 is None:
            m2_pos3 = db_m2_pos3

    logger.info(f"tel_status: {taken_at=},{inr=},{adc=},{m2_pos3=}")
    return taken_at, inr, adc, m2_pos3


def get_detected_objects(
    frame_id: int, filter_flags: int | None = BAD_DETECTION_FLAGS
) -> pd.DataFrame:
    """Get the detected objects from ``opdb.agc_data``.

    Parameters
    ----------
    frame_id : int
        The frame id of the frame.
    filter_flags : SourceDetectionFlag | int | None
        Bit-mask of flags to exclude.  Defaults to ``BAD_DETECTION_FLAGS``.

    Returns
    -------
    pd.DataFrame
        The detected objects (reset index).

    Raises
    ------
    RuntimeError
        If no valid detected objects remain after filtering.
    """
    logger.info("Getting detected objects from opdb.agc_data")
    detected_objects = query_agc_data(frame_id)
    logger.debug(f"Detected objects: {len(detected_objects)}")

    if filter_flags:
        logger.info(f"Filtering detected objects with bad flags: {filter_flags=}")
        detected_objects = detected_objects[(detected_objects['flags'] & int(filter_flags)) == 0]
        logger.debug(f"Detected objects after filtering bad flags: {len(detected_objects)=}")

    if len(detected_objects) == 0:
        raise RuntimeError("No valid spots detected, can't compute offsets")

    return detected_objects.reset_index(drop=True)


def search_gaia(ra, dec, radius=0.027 + 0.003):
    """Search guide stellar objects from Gaia DR3 sources.

    Parameters
    ----------
    ra : array_like
        The right ascensions (ICRS) of the search centers (deg).
    dec : array_like
        The declinations (ICRS) of the search centers (deg).
    radius : float
        The radius of the cones (deg).

    Returns
    -------
    astropy.table.Table
        The table of the Gaia DR3 sources inside the search areas.

    Raises
    ------
    RuntimeError
        If the Gaia DB query fails.
    """

    # Ensure inputs are iterable
    if np.isscalar(ra):
        ra = (ra,)
    if np.isscalar(dec):
        dec = (dec,)

    # Define columns and their units
    columns = (
        "source_id",
        "ref_epoch",
        "ra",
        "ra_error",
        "dec",
        "dec_error",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "phot_g_mean_mag",
    )

    units = (
        u.dimensionless_unscaled,
        u.yr,
        u.deg,
        u.mas,
        u.deg,
        u.mas,
        u.mas,
        u.mas,
        u.mas / u.yr,
        u.mas / u.yr,
        u.mas / u.yr,
        u.mas / u.yr,
        u.mag,
    )

    # Build query for all search centers
    radial_queries = [
        f"q3c_radial_query(ra,dec,{_ra},{_dec},{radius})" for _ra, _dec in zip(ra, dec)
    ]

    # Construct full SQL query
    query = (
        f"SELECT {','.join(columns)} FROM gaia3 WHERE "
        f"({' OR '.join(radial_queries)}) "
        f"AND pmra IS NOT NULL AND pmdec IS NOT NULL AND parallax IS NOT NULL "
        f"ORDER BY phot_g_mean_mag"
    )

    try:
        objects = GaiaDB().query_array(query)
    except Exception as e:
        raise RuntimeError(f"Failed to query Gaia DR3 sources: {e:r}")

    # Return results as an astropy Table
    return Table(rows=objects, names=columns, units=units)
