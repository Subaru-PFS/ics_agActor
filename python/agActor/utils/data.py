import logging
from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table
from ics.utils.database.db import DB
from ics.utils.database.gaia import GaiaDB
from ics.utils.database.opdb import OpDB
from numpy.typing import NDArray
from pfs.utils.datamodel.ag import AutoGuiderStarMask, SourceDetectionFlag

from agActor.catalog import astrometry, gen2_gaia as gaia
from agActor.catalog.pfs_design import pfsDesign as pfs_design
from agActor.utils.logging import log_message

logger = logging.getLogger(__name__)


DB_CLASSES = {"opdb": OpDB, "gaia": GaiaDB}


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
        guide_objects: DataFrame containing guide objects data
        ra: Right ascension of the field in degrees
        dec: Declination of the field in degrees
        inr: Instrument rotator angle in degrees
        inst_pa: Instrument position angle in degrees
        m2_pos3: M2 position 3 value in mm
        adc: Atmospheric dispersion corrector value
        taken_at: Time the frame was taken
    """

    guide_objects: pd.DataFrame
    ra: float
    dec: float
    inr: float
    inst_pa: float
    m2_pos3: float
    adc: float
    taken_at: Optional[datetime]
    guide_object_dtype: ClassVar[dict] = {
        "source_id": "<i8",  # u8 (80) not supported by FITSIO
        "ra": "<f8",
        "dec": "<f8",
        "mag": "<f4",
        "agc_camera_id": "<i4",
        "x": "<f4",
        "y": "<f4",
        "flags": "<i4",
    }

    def __str__(self) -> str:
        # Helper to safely format values
        def fmt(v, fmt_str="{:.3f}", unit=""):
            if v is None:
                return "None"
            try:
                return fmt_str.format(v) + unit
            except Exception:
                return str(v)

        # Field/telemetry info
        field = (
            f"Field: RA={fmt(self.ra, '{:.6f}')} deg, "
            f"Dec={fmt(self.dec, '{:.6f}')} deg, "
            f"INR={fmt(self.inr, '{:.3f}')} deg, "
            f"PA={fmt(self.inst_pa, '{:.3f}')} deg"
        )

        tel = (
            f"Tel: M2pos3={fmt(self.m2_pos3, '{:.3f}', ' mm')} "
            f"ADC={fmt(self.adc, '{:.3f}', ' deg')} "
            f"TakenAt={(self.taken_at.isoformat(timespec='seconds') if isinstance(self.taken_at, datetime) else str(self.taken_at))}"
        )

        # Guide objects summary
        n = 0
        mag_part = "mag=NA"
        cam_part = "cams=NA"
        filt_part = "filtered=NA"

        try:
            if isinstance(self.guide_objects, pd.DataFrame) and not self.guide_objects.empty:
                df = self.guide_objects
                n = len(df)

                # Magnitude stats if available
                if "mag" in df.columns:
                    mags = pd.to_numeric(df["mag"], errors="coerce").dropna()
                    if len(mags) > 0:
                        mag_part = (
                            f"mag=[{mags.min():.2f},{mags.median():.2f},{mags.max():.2f}]"
                        )
                    else:
                        mag_part = "mag=NA"

                # Camera distribution if available
                if "agc_camera_id" in df.columns:
                    cams = df["agc_camera_id"].dropna()
                    try:
                        cams = cams.astype(int)
                    except Exception:
                        pass
                    if len(cams) > 0:
                        unique = sorted(pd.unique(cams).tolist())
                        cam_part = f"cams={unique}"

                # Filtered count if column exists (from filter_guide_objects)
                if "filtered_by" in df.columns:
                    n_filtered = int((df["filtered_by"] != 0).sum())
                    filt_part = f"filtered={n_filtered}/{n}"
                else:
                    filt_part = f"count={n}"
            else:
                cam_part = "cams=[]"
                mag_part = "mag=NA"
                filt_part = "count=0"
        except Exception:
            # Be defensive: never raise from __str__
            cam_part = "cams=?"
            mag_part = "mag=?"
            filt_part = f"count={n if n else '?'}"

        objs = f"GuideObjects: count={n}, {mag_part}, {cam_part}, {filt_part}"

        return " | ".join([field, tel, objs])



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
    guide_objects: pd.DataFrame
    detected_objects: pd.DataFrame
    identified_objects: NDArray
    dx: float
    dy: float
    size: float
    peak: float
    flux: float
    design_id: int | None = None
    visit_id: int | None = None
    frame_id: int | None = None

    def __str__(self) -> str:
        # Helper to format optional floats
        def fmt_opt(v: Optional[float], unit: str = "") -> str:
            if v is None:
                return "None"
            return f"{v:.3f}{unit}"

        # Counts for arrays (avoid dumping arrays themselves)
        n_guide = 0 if self.guide_objects is None else int(len(self.guide_objects))
        n_detected = 0 if self.detected_objects is None else int(len(self.detected_objects))
        n_matched = 0 if self.identified_objects is None else int(len(self.identified_objects))

        parts = [
            f"Frame: frame_id={self.frame_id} visit_id={self.visit_id} design_id={self.design_id}",
            f"Field: RA={self.ra:.6f} deg, Dec={self.dec:.6f} deg, PA={self.inst_pa:.3f} deg",
            (
                "Offsets: "
                f'dRA={self.ra_offset:.3f}" '
                f'dDec={self.dec_offset:.3f}" '
                f'dINR={self.inr_offset:.3f}" '
                f"dScale={self.scale_offset:.6f} "
                f"dAlt={fmt_opt(self.dalt, '"')} "
                f"dAz={fmt_opt(self.daz, '"')} "
                f'dx={self.dx:.3f}" dy={self.dy:.3f}"'
            ),
            f"Spot: size={self.size:.3f}, peak={self.peak:.1f}, flux={self.flux:.1f}",
            f"Counts: guide={n_guide}, detected={n_detected}, matched={n_matched}",
        ]
        return " | ".join(parts)


class GuideOffsetFlag(IntFlag):
    """Flags for the guide offsets.
    Attributes:
        OK: Guide offset is OK.
        INVALID_OFFSET: Guide offset is invalid and was not used.
    """

    OK = 0x0000
    INVALID_OFFSET = 0x0001


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
        _, visit_id, _, _, db_taken_at, _, _, db_inr, db_adc, _, _, _, db_m2_pos3 = (
            query_agc_exposure(frame_id, as_dataframe=False)
        )

        # If sequence_id is provided, get more accurate information from tel_status table
        sequence_id = kwargs.get("sequence_id")
        if sequence_id is not None:
            logger.debug(
                f"Getting telescope status from opdb for {visit_id=},{sequence_id=}"
            )
            _, _, db_inr, db_adc, db_m2_pos3, _, _, _, _, db_taken_at = (
                query_tel_status(visit_id, sequence_id, as_dataframe=False)
            )

        # Use database values for any missing parameters
        taken_at = taken_at or db_taken_at
        inr = inr or db_inr
        adc = adc or db_adc
        m2_pos3 = m2_pos3 or db_m2_pos3

    logger.debug(f"{taken_at=},{inr=},{adc=},{m2_pos3=}")
    return taken_at, inr, adc, m2_pos3


def get_guide_objects(
    frame_id=None, obswl: float = 0.62, **kwargs
) -> GuideObjectsResult:
    """Get the guide objects for a given frame or from other sources.

    The guide objects can come from four separate sources:
    1. REF_SKY: If frame_id is provided and detected_objects are passed, guide objects are generated using astrometry.measure.
    2. REF_OTF: If neither design_id nor design_path is provided, guide objects are fetched from the Gaia database.
    3. REF_DB:  If design_id and design_path are provided, guide objects are fetched from the specified PFS design file.
    4. REF_DB:  If only design_id is provided, guide objects are fetched from the operational database (opdb).

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
    # Use logger.debug if logger is None, otherwise use log_message
    log_fn = lambda msg: log_message(logger, msg)

    # Get telescope status if frame_id is provided
    if frame_id is not None:
        taken_at, inr, adc, m2_pos3 = get_telescope_status(frame_id=frame_id, **kwargs)
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

    # Check if we should use astrometry.measure with detected objects, i.e. REF_SKY.
    detected_objects = kwargs.get("detected_objects")
    if frame_id is not None and detected_objects is not None:
        log_fn("Getting guide objects from astrometry")
        guide_object_rows = astrometry.measure(
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
        log_fn(f"Got {len(guide_object_rows)=} guide objects")
        guide_objects = pd.DataFrame(
            guide_object_rows,
            columns=list(GuideObjectsResult.guide_object_dtype.keys()),
        )

    # Check if we should use the gaia catalog, i.e. REF_OTF.
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
        guide_object_rows = gaia.process_search_results(
            _objects,
            frame_tc.obstime,
            altaz_c,
            frame_tc,
            adc,
            inr,
            m2pos3=m2_pos3,
            obswl=obswl,
        )

        guide_objects = pd.DataFrame(
            guide_object_rows,
            columns=list(GuideObjectsResult.guide_object_dtype.keys()),
        )

    # Check if we should use the design, either from the file or the opdb, i.e. REF_DB.
    else:
        if design_path is not None:
            log_fn(
                f"Getting guide_objects via pfsDesign file at '{design_path}{design_id}'"
            )
            guide_objects, _ra, _dec, _inst_pa = pfs_design(
                design_id, design_path, logger=logger
            ).guide_objects(obstime=taken_at)
        else:
            log_fn(f"Getting guide_objects from opdb via {design_id}")
            _, _ra, _dec, _inst_pa, *_ = query_pfs_design(design_id)
            guide_objects = query_pfs_design_agc(design_id, as_dataframe=True)

        # Rename columns for consistency.
        new_cols = dict(
            zip(guide_objects.columns, GuideObjectsResult.guide_object_dtype.keys())
        )
        guide_objects = guide_objects.rename(columns=new_cols)

        # Mark which guide objects should be filtered (only GALAXIES for now).
        guide_objects = filter_guide_objects(guide_objects)

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
) -> pd.DataFrame:
    """Get the detected objects from opdb.agc_data.

    Parameters:
    -----------
        frame_id: int
            The frame id of the frame.
        filter_flag: SourceDetectionFlag | int
            Flags greater than or equal to this flag will be filtered.

    Returns:
    --------
        pd.DataFrame:
            The detected objects.

    Raises:
    -------
    RuntimeError:
        If no detected objects are found.
    """
    logger.info("Getting detected objects from opdb.agc_data")
    detected_objects = query_agc_data(frame_id)
    logger.debug(f"Detected objects: {len(detected_objects)=}")

    if filter_flag:
        detected_objects = detected_objects.query(f"flags < {filter_flag}")
        logger.debug(f"Detected objects after filtering: {len(detected_objects)=}")

    if len(detected_objects) == 0:
        raise RuntimeError("No valid spots detected, can't compute offsets")

    return detected_objects.reset_index(drop=True)


def write_agc_guide_offset(
    *,
    frame_id: int,
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
):
    """Write the guide offsets to the database.

    If a value is not passed to the function then the default `None` will be
    written to the database.

    Parameters:
        frame_id (int): The frame id of the frame.
        ra (float | None): Right ascension of the field in degrees.
        dec (float | None): Declination of the field in degrees.
        pa (float | None): Instrument position angle in degrees.
        delta_ra (float | None): Right ascension offset in arcseconds.
        delta_dec (float | None): Declination offset in arcseconds.
        delta_insrot (float | None): Instrument rotator offset in arcseconds.
        delta_scale (float | None): Scale change.
        delta_el (float | None): Elevation (Altitude) offset in arcseconds.
        delta_az (float | None): Azimuth offset in arcseconds.
        delta_z (float | None): Average focus offset.
        delta_zs (NDArray | None): Focus offset per camera.
        offset_flags (GuideOffsetFlag): Any flags for the data, stored in
            the `mask` column, defaults to `GuideOffsetFlag.OK`.
    """
    params = dict(
        agc_exposure_id=frame_id,
        guide_ra=ra,
        guide_dec=dec,
        guide_pa=pa,
        guide_delta_ra=float(delta_ra),
        guide_delta_dec=float(delta_dec),
        guide_delta_insrot=float(delta_insrot),
        guide_delta_scale=float(delta_scale),
        guide_delta_az=float(delta_az),
        guide_delta_el=float(delta_el),
        mask=offset_flags.value,
        guide_delta_z=float(delta_z),
    )
    if delta_zs is not None:
        params.update(guide_delta_z1=float(delta_zs[0]))
        params.update(guide_delta_z2=float(delta_zs[1]))
        params.update(guide_delta_z3=float(delta_zs[2]))
        params.update(guide_delta_z4=float(delta_zs[3]))
        params.update(guide_delta_z5=float(delta_zs[4]))
        params.update(guide_delta_z6=float(delta_zs[5]))


    logger.info(f"Writing agc_guide_offsets with {params=}")
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
            "agc_camera_id": int(detected_objects["agc_camera_id"][detected_idx]),
            "spot_id": int(detected_objects["spot_id"][detected_idx]),
            "guide_star_id": int(guide_objects["source_id"][guide_idx]),
            "agc_nominal_x_mm": float(nominal_x_mm),
            "agc_nominal_y_mm": float(nominal_y_mm),
            "agc_center_x_mm": float(center_x_mm),
            "agc_center_y_mm": float(center_y_mm),
            "flags": int(guide_objects["flags"][guide_idx]),
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


def query_db(
    sql: str,
    params: tuple | None = None,
    as_dataframe: bool = True,
    db: DB | None = None,
) -> pd.DataFrame | np.ndarray | None:
    """Helper method to return rows from the sql query either.

    Parameters
    ----------
    sql : str
        The sql query to execute.
    params : tuple
        The parameters to pass to the sql query.
    as_dataframe : bool
        Whether to return a pandas dataframe from the sql query.
        Defaults to True.
    db : DB | None
        The database to use. Defaults to None, which uses `get_db("opdb").

    Returns
    -------
    pd.DataFrame | np.ndarray | None
        The results from the query.
    """
    db = db or get_db("opdb")
    if as_dataframe:
        result = pd.read_sql(sql, db.engine, params=params)
        if len(result) == 1:
            result = result.iloc[0]
    else:
        result = db.fetchone(query=sql, params=params)
        if len(result) == 1:
            result = result[0]

    return result


def query_agc_data(agc_exposure_id: int, as_dataframe: bool = True, **kwargs):
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
    params = (agc_exposure_id,)
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_tel_status(
    pfs_visit_id: int, status_sequence_id: int, as_dataframe: bool = True, **kwargs
):
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
    params = (pfs_visit_id, status_sequence_id)
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_agc_exposure(agc_exposure_id: int, as_dataframe: bool = True, **kwargs):
    sql = """
SELECT
    t0.agc_exposure_id,
    t0.pfs_visit_id,
    t1.pfs_design_id,
    t0.agc_exptime,
    t0.taken_at,
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
    t0.agc_exposure_id=%s
"""
    params = (agc_exposure_id,)
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_pfs_design_agc(pfs_design_id: int, as_dataframe: bool = True, **kwargs):
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
    params = (pfs_design_id,)
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def query_pfs_design(pfs_design_id: int, as_dataframe: bool = True, **kwargs):
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
    params = (pfs_design_id,)
    return query_db(sql, params, as_dataframe=as_dataframe, **kwargs)


def filter_guide_objects(
    guide_objects, is_acquisition=True, flag_column="flags"
) -> pd.DataFrame:
    """Apply filtering to the guide objects based on their flags.

    This function filters guide objects based on various quality flags. For initial
    coarse guiding, it only filters out galaxies. For fine guiding (initial=False),
    it applies additional filters to ensure high quality guide stars from GAIA.

    Parameters
    ----------
    guide_objects : pd.DataFrame
        Structured array containing guide object data including flags.
        Must have fields for basic star data (objId, ra, dec, mag, etc.)
        and optionally a 'flag' field for filtering.
    is_acquisition : bool, optional
        If True, only filter galaxies. If False (default), apply all quality filters
        including GAIA star requirements.
    flag_column : str, optional
        Indicates the column name that includes the filter flags.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with an added `filtered_by` column.
    """
    guide_objects_df = guide_objects.copy()
    guide_objects_df["filtered_by"] = 0

    # Filter out the galaxies.
    logger.info("Filtering galaxies from results.")
    galaxy_idx = (guide_objects_df[flag_column].values & AutoGuiderStarMask.GALAXY) != 0
    guide_objects_df.loc[galaxy_idx, "filtered_by"] = AutoGuiderStarMask.GALAXY.value

    if is_acquisition:
        filters_for_inclusion = [
            AutoGuiderStarMask.GAIA,
            AutoGuiderStarMask.NON_BINARY,
            AutoGuiderStarMask.ASTROMETRIC,
            AutoGuiderStarMask.PMRA_SIG,
            AutoGuiderStarMask.PMDEC_SIG,
            AutoGuiderStarMask.PARA_SIG,
            AutoGuiderStarMask.PHOTO_SIG,
        ]

        # Go through the filters and mark which stars would be flagged as NOT meeting the mask requirement.
        for f in filters_for_inclusion:
            not_filtered = guide_objects_df.filtered_by != AutoGuiderStarMask.GALAXY
            include_filter = (guide_objects_df[flag_column].values & f) == 0
            to_be_filtered = (include_filter & not_filtered) != 0
            guide_objects_df.loc[to_be_filtered, "filtered_by"] |= f.value
            logger.info(
                f"Filtering by {f.name}, removes {to_be_filtered.sum()} guide objects."
            )

    return guide_objects_df
