import logging
from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from ics.utils.database.db import DB
from ics.utils.database.gaia import GaiaDB
from ics.utils.database.opdb import OpDB
from numpy.typing import NDArray

from agActor.catalog import astrometry, gen2_gaia as gaia
from agActor.catalog.pfs_design import pfsDesign as pfs_design
from agActor.utils.logging import log_message

logger = logging.getLogger(__name__)


DB_CLASSES = {'opdb': OpDB, 'gaia': GaiaDB}

class Database:
    def __init__(self):
        self.dbs: dict[str, DB | None] = {"opdb": None, "gaia": None}

    def setup_db(self, dbname: str, dsn: str | dict | None = None, **kwargs) -> DB:
        """Sets up the database.

        Parameters
        ----------
        dbname : str
            Database name, either "opdb" or "gaia".
        dsn : str or dict
            Database connection parameters. Can be anything accepted by
            `ics.utils.database.opdb.DB`, namely a string or dict of
            connection parameters.
        **kwargs : dict
            Additional keyword arguments to pass to OpDB constructor
        """
        if self.dbs[dbname] is None:
            self.dbs[dbname] = DB_CLASSES[dbname](dsn=dsn, **kwargs)
            logger.info(f"Created database {dbname} with {dsn=}")
        else:
            logger.debug(f"Database {dbname} already setup: {self.dbs[dbname].dsn}")

        return self.dbs[dbname]

    def get_db(self, dbname: str) -> DB | None:
        """Returns the database object for the given database name.

        Parameters
        ----------
        dbname : str
            Database name, either "opdb" or "gaia".

        Returns
        -------
        DB | None
            Database object for the given database name otherwise None.
        """
        try:
            db = self.setup_db(dbname=dbname)
            logger.debug(f"Returning database {dbname}: {db.dsn}")
            return db
        except KeyError:
            logger.warning(f"Database {dbname} not found.")
            return None


# Create the global database holder.
_DATABASE = Database()


def setup_db(dbname: str, dsn: str | dict, **kwargs):
    """Sets up the specific database using the global Database instance.

    Parameters
    ----------
    dbname : str
        Database name, either "opdb" or "gaia".
    dsn : str or dict
        Database connection parameters. Can be anything accepted by
        `ics.utils.database.opdb.DB`, namely a string or dict of
        connection parameters.
    **kwargs : dict
        Additional keyword arguments to pass to OpDB constructor
    """
    _DATABASE.setup_db(dbname=dbname, dsn=dsn, **kwargs)


def get_db(dbname: str) -> DB | None:
    """Returns the database object for the given database name.

    Parameters
    ----------
    dbname : str
        Database name, either "opdb" or "gaia".

    Returns
    -------
    db : DB | None
        Database object for the given database name otherwise None.
    """
    db = _DATABASE.get_db(dbname=dbname)
    if db is None:
        logger.warning(f"Database {dbname} not found.")
        return None

    return db


@dataclass
class GuideObjectsResult:
    """Result of the get_guide_objects function.

    Attributes:
        guide_objects: Structured array containing guide objects data
        ra: Right ascension of the field in degrees
        dec: Declination of the field in degrees
        inr: Instrument rotator angle in degrees
        inst_pa: Instrument position angle in degrees
        m2_pos3: M2 position 3 value in mm
        adc: Atmospheric dispersion corrector value
        taken_at: Time the frame was taken
    """

    guide_objects: np.ndarray
    ra: float
    dec: float
    inr: float
    inst_pa: float
    m2_pos3: float
    adc: float
    taken_at: Optional[datetime]


@dataclass
class GuideOffsets:
    """Result of the autoguide function.

    Attributes:
        ra: Right ascension of the field in degrees
        dec: Declination of the field in degrees
        inst_pa: Instrument position angle in degrees
        ra_offset: Right ascension offset in arcseconds
        dec_offset: Declination offset in arcseconds
        inr_offset: Instrument rotator offset in arcseconds
        scale_offset: Scale offset
        dalt: Altitude offset in arcseconds
        daz: Azimuth offset in arcseconds
        guide_objects: Guide objects used for calculations
        detected_objects: Detected objects from the frame
        identified_objects: Matched guide and detected objects
        dx: X offset in arcseconds
        dy: Y offset in arcseconds
        size: Representative spot size
        peak: Representative peak intensity
        flux: Representative flux
    """

    ra: float
    dec: float
    inst_pa: float
    ra_offset: float
    dec_offset: float
    inr_offset: float
    scale_offset: float
    dalt: Optional[float]
    daz: Optional[float]
    guide_objects: NDArray
    detected_objects: NDArray
    identified_objects: NDArray
    dx: float
    dy: float
    size: float
    peak: float
    flux: float


# TODO move this somewhere else.
class SourceDetectionFlag(IntFlag):
    """
    Represents a bitmask for detection properties.

    Attributes:
        RIGHT: Source is detected on the right side of the image.
        EDGE: Source is detected at the edge of the image.
        SATURATED: Source is saturated.
        BAD_SHAPE: Source has a bad shape.
        BAD_ELLIP: Source has a bad ellipticity.
        FLAT_TOP: Source has a flat top profile.
    """

    RIGHT = 0x0001
    EDGE = 0x0002
    SATURATED = 0x0004
    BAD_SHAPE = 0x0008
    BAD_ELLIP = 0x0010
    FLAT_TOP = 0x0020


def get_telescope_status(*, frame_id, **kwargs):
    """Get the telescope status information for a specific frame ID.

    Args:
        frame_id: The frame ID to retrieve telescope status for.
        **kwargs: Additional keyword arguments that may include:
            taken_at: Timestamp when the frame was taken
            inr: Instrument rotator angle
            adc: Atmospheric dispersion corrector value
            m2_pos3: Secondary mirror position
            sequence_id: Sequence ID for querying telescope status

    Returns:
        tuple: A tuple containing:
            taken_at: Timestamp when the frame was taken
            inr: Instrument rotator angle in degrees
            adc: Atmospheric dispersion corrector value
            m2_pos3: Secondary mirror position in mm

    If any of the optional values are not provided in kwargs, they will be retrieved
    from the opdb database using the frame_id and sequence_id (if provided).
    """
    logging.debug(f"Getting telescope status for {frame_id=}")

    # Extract values from kwargs if provided
    taken_at = kwargs.get("taken_at")
    inr = kwargs.get("inr")
    adc = kwargs.get("adc")
    m2_pos3 = kwargs.get("m2_pos3")

    # Check if we need to fetch any missing values from the database
    if any(value is None for value in (taken_at, inr, adc, m2_pos3)):
        # First, query the agc_exposure table to get basic information, including visit_id.
        logging.debug(f"Getting agc_exposure from opdb for frame_id={frame_id}")
        visit_id, _, db_taken_at, _, _, db_inr, db_adc, _, _, _, db_m2_pos3 = (
            query_agc_exposure(frame_id)
        )

        # If sequence_id is provided, get more accurate information from tel_status table
        sequence_id = kwargs.get("sequence_id")
        if sequence_id is not None:
            logging.debug(
                f"Getting telescope status from opdb for {visit_id=},{sequence_id=}"
            )
            _, _, db_inr, db_adc, db_m2_pos3, _, _, _, _, db_taken_at = (
                query_tel_status(visit_id, sequence_id)
            )

        # Use database values for any missing parameters
        taken_at = taken_at or db_taken_at
        inr = inr or db_inr
        adc = adc or db_adc
        m2_pos3 = m2_pos3 or db_m2_pos3

    logging.debug(f"{taken_at=},{inr=},{adc=},{m2_pos3=}")
    return taken_at, inr, adc, m2_pos3


def get_guide_objects(
    frame_id=None,
    obswl: float = 0.62,
    logger=None,
    **kwargs,
):
    """Get the guide objects for a given frame or from other sources.

    The guide objects can come from four separate sources:
    1. If frame_id is provided and detected_objects are passed, guide objects are generated using astrometry.measure
    2. If neither design_id nor design_path is provided, guide objects are fetched from the Gaia database.
    3. If design_path is provided, guide objects are fetched from the specified PFS design file.
    4. If only design_id is provided, guide objects are fetched from the operational database (opdb).

    Parameters:
        frame_id (int, optional): The frame id of the frame. If None, telescope status is taken from kwargs.
        obswl (float): The observation wavelength in microns, by default 0.62.
        logger (Logger, optional): Logger for logging messages.
        **kwargs: Additional keyword arguments including:
            detected_objects: If provided with frame_id, used to generate guide objects with astrometry.measure
            design_id: PFS design ID
            design_path: Path to PFS design file
            ra, dec, inst_pa: Field center coordinates and position angle
            taken_at, inr, adc, m2_pos3: Telescope status parameters

    Returns:
        GuideObjectsResult: A dataclass containing:
            guide_objects (np.ndarray): The guide objects.
            ra (float): The right ascension of the field.
            dec (float): The declination of the field.
            inr (float): The instrument rotator angle.
            inst_pa (float): The instrument position angle.
            m2_pos3 (float): The M2 position 3 value.
            adc (float): The ADC setting.
            taken_at (datetime): The time the frame was taken.
    """
    # Use logging.debug if logger is None, otherwise use log_message
    log_fn = lambda msg: log_message(logger, msg)

    # Get telescope status if frame_id is provided
    if frame_id is not None:
        taken_at, inr, adc, m2_pos3 = get_telescope_status(
            frame_id=frame_id, logger=logger, **kwargs
        )
    else:
        # Extract telescope status from kwargs
        taken_at = kwargs.get("taken_at")
        inr = kwargs.get("inr")
        adc = kwargs.get("adc")
        m2_pos3 = kwargs.get("m2_pos3", 6.0)
        log_fn(f"taken_at={taken_at},inr={inr},adc={adc},m2_pos3={m2_pos3}")

    design_id = kwargs.get("design_id")
    design_path = kwargs.get("design_path")
    log_fn(f"design_id={design_id},design_path={design_path}")
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")

    # Apply coordinate adjustments if provided
    if "dra" in kwargs and ra is not None:
        ra += kwargs.get("dra") / 3600
    if "ddec" in kwargs and dec is not None:
        dec += kwargs.get("ddec") / 3600
    if "dinr" in kwargs and inr is not None:
        inr += kwargs.get("dinr") / 3600

    # Check if we should use astrometry.measure with detected objects
    detected_objects = kwargs.get("detected_objects")
    if frame_id is not None and detected_objects is not None:
        log_fn("Getting guide objects from astrometry")
        guide_objects = astrometry.measure(
            detected_objects=detected_objects,
            ra=ra,
            dec=dec,
            obstime=taken_at,
            inst_pa=inst_pa,
            adc=adc,
            m2_pos3=m2_pos3,
            obswl=obswl,
            logger=logger,
        )
        log_fn(f"Got {len(guide_objects)=} guide objects")
    elif design_path is None and design_id is None:
        log_fn(
            "No design_id or design_path provided, getting guide objects from gaia db."
        )
        # Set up search coordinates first
        icrs, altaz_c, frame_tc, inr, adc = gaia.setup_search_coordinates(
            ra=ra,
            dec=dec,
            obstime=taken_at,
            inst_pa=inst_pa,
            adc=adc,
            m2pos3=m2_pos3,
            obswl=obswl,
        )

        # Search for objects
        _objects = search_gaia(icrs.ra.deg, icrs.dec.deg)

        # Process search results and get structured array directly
        guide_objects = gaia.process_search_results(
            _objects,
            frame_tc.obstime,
            altaz_c,
            frame_tc,
            adc,
            inr,
            m2pos3=m2_pos3,
            obswl=obswl,
        )
    else:
        if design_path is not None:
            log_fn(f"Getting guide_objects via {design_path}")
            guide_objects, _ra, _dec, _inst_pa = pfs_design(
                design_id, design_path, logger=logger
            ).guide_objects(obstime=taken_at)
        else:
            log_fn(f"Getting guide_objects from opdb via {design_id}")
            _, _ra, _dec, _inst_pa, *_ = query_pfs_design(design_id)
            guide_objects = query_pfs_design_agc(design_id)

        ra = ra or _ra
        dec = dec or _dec
        inst_pa = inst_pa or _inst_pa

    return GuideObjectsResult(
        guide_objects=guide_objects,
        ra=ra,
        dec=dec,
        inr=inr,
        inst_pa=inst_pa,
        m2_pos3=m2_pos3,
        adc=adc,
        taken_at=taken_at,
    )


def get_detected_objects(
    frame_id: int, filter_flag: int = SourceDetectionFlag.EDGE
) -> np.ndarray:
    """Get the detected objects from opdb.agc_data.

    Parameters:
        frame_id (int): The frame id of the frame.
        filter_flag (SourceDetectionFlag | int): The flag used to determine if an object is detected, by default SourceDetectionFlag.EDGE.

    Returns:
        np.ndarray: The detected objects.

    """
    logging.debug("Getting detected objects from opdb.agc_data")
    detected_objects_rows = query_agc_data(frame_id)

    detected_objects_dtype = [
        ("camera_id", np.int16),
        ("spot_id", np.int16),
        ("moment_00", np.float32),
        ("centroid_x", np.float32),
        ("centroid_y", np.float32),
        ("central_moment_11", np.float32),
        ("central_moment_20", np.float32),
        ("central_moment_02", np.float32),
        ("peak_x", np.uint16),
        ("peak_y", np.uint16),
        ("peak", np.uint16),
        ("background", np.float32),
        ("flags", np.uint8),
    ]

    detected_objects = np.rec.fromarrays(
        detected_objects_rows.T,
        dtype=detected_objects_dtype,
    )

    logging.debug(f"Detected objects: {len(detected_objects)=}")

    if filter_flag:
        detected_objects = detected_objects[
            detected_objects["flags"] <= SourceDetectionFlag.RIGHT
        ]
        logging.debug(
            f"Detected objects after source filtering: {len(detected_objects)=}"
        )

    if len(detected_objects) == 0:
        raise RuntimeError("No valid spots detected, can't compute offsets")

    return detected_objects


def write_agc_guide_offset(
    *,
    frame_id,
    ra=None,
    dec=None,
    pa=None,
    delta_ra=None,
    delta_dec=None,
    delta_insrot=None,
    delta_scale=None,
    delta_az=None,
    delta_el=None,
    delta_z=None,
    delta_zs=None,
):
    params = dict(
        agc_exposure_id=frame_id,
        guide_ra=ra,
        guide_dec=dec,
        guide_pa=pa,
        guide_delta_ra=delta_ra,
        guide_delta_dec=delta_dec,
        guide_delta_insrot=delta_insrot,
        guide_delta_scale=delta_scale,
        guide_delta_az=delta_az,
        guide_delta_el=delta_el,
        guide_delta_z=delta_z,
    )
    if delta_zs is not None:
        params.update(guide_delta_z1=delta_zs[0])
        params.update(guide_delta_z2=delta_zs[1])
        params.update(guide_delta_z3=delta_zs[2])
        params.update(guide_delta_z4=delta_zs[3])
        params.update(guide_delta_z5=delta_zs[4])
        params.update(guide_delta_z6=delta_zs[5])

    get_db("opdb").insert("agc_guide_offset", **params)

def write_agc_match(
    *,
    design_id: int,
    frame_id: int,
    guide_objects: NDArray,
    detected_objects: NDArray,
    identified_objects: NDArray,
) -> int | None:
    """Insert AG identified objects into opdb.agc_match.

    Parameters:
    -----------
    design_id (int): The PFS design ID.
    frame_id (int): The exposure ID for the AGC frame.
    guide_objects (NDArray): Dictionary or structured array containing guide star data.
    detected_objects (NDArray): Dictionary or structured array containing detected object data.
    identified_objects (NDArray): An iterable of tuples, where each tuple contains
                                   indices and coordinate data for a matched object.
                                   Expected format: (detected_idx, guide_idx,
                                   center_x, center_y, nominal_x, nominal_y, ...)

    Returns:
    --------
    int | None
        The number of identified objects inserted or None if no matches.
    """
    rows_to_insert = []
    for match in identified_objects:
        detected_idx = match[0]
        guide_idx = match[1]
        center_x_mm = match[2]
        center_y_mm = match[3] * -1  # TODO: move negative, see INSTRM-2654
        nominal_x_mm = match[4]
        nominal_y_mm = match[5] * -1  # TODO: move negative, see INSTRM-2654

        row = {
            "pfs_design_id": design_id,
            "agc_exposure_id": frame_id,
            "agc_camera_id": int(detected_objects["camera_id"][detected_idx]),
            "spot_id": int(detected_objects["spot_id"][detected_idx]),
            "guide_star_id": int(guide_objects["source_id"][guide_idx]),
            "agc_nominal_x_mm": float(nominal_x_mm),
            "agc_nominal_y_mm": float(nominal_y_mm),
            "agc_center_x_mm": float(center_x_mm),
            "agc_center_y_mm": float(center_y_mm),
            "flags": int(guide_objects["filter_flag"][guide_idx]),
        }
        rows_to_insert.append(row)

    if rows_to_insert:
        df = pd.DataFrame(rows_to_insert)
        logger.debug("Inserting data into database")
        n_rows = get_db("opdb").insert_dataframe(df=df, table="agc_match")
        logger.info(f"Finished inserting agc_match data: {n_rows} rows inserted")

        return n_rows

    return None


def search_gaia(ra, dec, radius=0.027 + 0.003):
    """
    Search guide stellar objects from Gaia DR3 sources.

    Parameters
    ----------
    ra : array_like
        The right ascensions (ICRS) of the search centers (deg)
    dec : array_like
        The declinations (ICRS) of the search centers (deg)
    radius : scalar
        The radius of the cones (deg)

    Returns
    -------
    astropy.table.Table
        The table of the Gaia DR3 sources inside the search areas
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
        objects = get_db("gaia").fetchall(query)
    except Exception as e:
        raise RuntimeError(f"Failed to query Gaia DR3 sources: {e:r}")

    # Return results as an astropy Table
    return Table(rows=objects, names=columns, units=units)


def query_agc_data(agc_exposure_id):
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
WHERE agc_exposure_id = %s
ORDER BY agc_camera_id, spot_id
"""
    return get_db("opdb").fetchall(sql, (agc_exposure_id,))


def query_tel_status(pfs_visit_id, status_sequence_id):
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
created_at
FROM tel_status
WHERE pfs_visit_id=%s AND status_sequence_id=%s
"""
    return get_db("opdb").fetchone(
        sql,
        (
            pfs_visit_id,
            status_sequence_id,
        ),
    )


def query_agc_exposure(agc_exposure_id):
    sql = """
SELECT
pfs_visit_id,
agc_exptime,
taken_at,
azimuth,
altitude,
insrot,
adc_pa,
outside_temperature,
outside_humidity,
outside_pressure,
m2_pos3
FROM agc_exposure
WHERE agc_exposure_id=%s
"""
    return get_db("opdb").fetchone(sql, (agc_exposure_id,))


def query_pfs_design_agc(pfs_design_id):
    sql = """
SELECT
guide_star_id,
guide_star_ra,
guide_star_dec,
guide_star_magnitude,
agc_camera_id,
agc_target_x_pix,
agc_target_y_pix,
guide_star_flag
FROM pfs_design_agc
WHERE pfs_design_id=%s
ORDER BY guide_star_id
"""
    return get_db("opdb").fetchall(sql, (pfs_design_id,))


def query_pfs_design(pfs_design_id):
    sql = """
SELECT
tile_id,
ra_center_designed,
dec_center_designed,
pa_designed,
num_sci_designed,
num_cal_designed,
num_sky_designed,
num_guide_stars,
exptime_tot,
exptime_min,
ets_version,
ets_assigner,
designed_at,
to_be_observed_at,
is_obsolete
FROM pfs_design
WHERE pfs_design_id=%s
"""
    return get_db("opdb").fetchone(sql, (pfs_design_id,))
