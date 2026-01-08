import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntFlag
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import Table

from numpy.typing import NDArray
from pfs.utils.coordinates import updateTargetPosition
from pfs.utils.coordinates.CoordTransp import ag_pfimm_to_pixel

from pfs.utils.database.db import DB
from pfs.utils.database.opdb import OpDB
from pfs.utils.database.gaia import GaiaDB

from pfs.utils.datamodel.ag import AutoGuiderStarMask, SourceDetectionFlag

logger = logging.getLogger(__name__)


BAD_DETECTION_FLAGS = (
    SourceDetectionFlag.SATURATED
    | SourceDetectionFlag.EDGE
    | SourceDetectionFlag.BAD_ELLIP
    | SourceDetectionFlag.FLAT_TOP
)


@dataclass
class GuideCatalog:
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
        "pm_ra": "<f8",
        "pm_dec": "<f8",
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
        filt_part = "count=NA"

        try:
            if (
                isinstance(self.guide_objects, pd.DataFrame)
                and not self.guide_objects.empty
            ):
                df = self.guide_objects
                n = len(df)

                # Magnitude stats if available
                if "mag" in df.columns:
                    mags = pd.to_numeric(df["mag"], errors="coerce").dropna()
                    if len(mags) > 0:
                        mag_part = f"mag=[{mags.min():.2f},{mags.median():.2f},{mags.max():.2f}]"
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
                    filt_part = f"count={n - n_filtered} filtered={n_filtered}"
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

        objs = f"GuideObjects: {filt_part}, {mag_part}, {cam_part}"

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
    identified_objects: pd.DataFrame
    dx: float
    dy: float
    size: float
    peak: float
    flux: float
    design_id: int | None = None
    visit_id: int | None = None
    frame_id: int | None = None

    def save_numpy_files(self, base_dir: str = "/dev/shm") -> list:
        """Save guide, detected, and identified objects to numpy files.

        This is a bit verbose but guarantees consistent formatting with what
        the other actors expect.

        Parameters:
        -----------
        base_dir : str
            Directory to save files in, defaults to /dev/shm/.

        Returns:
        --------
        full_files : list
            List of saved files.
        """
        save_files = {}

        # Guide objects.
        guide_npy = np.array(
            [
                (
                    row.source_id,
                    row.ra,
                    row.dec,
                    row.mag,
                    row.agc_camera_id,
                    row.x,
                    row.y,
                    row.x_dp,
                    row.y_dp,
                    row.flags,
                    row.filtered_by,
                )
                for row in self.guide_objects.itertuples(index=False)
            ],
            dtype=[
                ("source_id", np.int64),  # u8 (80) not supported by FITSIO
                ("ra", np.float64),
                ("dec", np.float64),
                ("mag", np.float32),
                ("camera_id", np.int16),
                ("x", np.float32),
                ("y", np.float32),
                ("x_dp", np.float32),
                ("y_dp", np.float32),
                ("flags", np.int16),
                ("filter_flag", np.uint16),
            ],
        )
        save_files["guide_objects"] = guide_npy

        # Detected objects.
        detected_npy = np.array(
            [
                (
                    row.agc_camera_id,
                    row.spot_id,
                    row.image_moment_00_pix,
                    row.centroid_x_pix,
                    row.centroid_y_pix,
                    row.central_image_moment_11_pix,
                    row.central_image_moment_20_pix,
                    row.central_image_moment_02_pix,
                    row.peak_pixel_x_pix,
                    row.peak_pixel_y_pix,
                    row.peak_intensity,
                    row.background,
                    row.flags,
                )
                for row in self.detected_objects.itertuples(index=False)
            ],
            dtype=[
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
            ],
        )
        save_files["detected_objects"] = detected_npy

        # Identified objects.
        ident_npy = np.array(
            [
                (
                    row.detected_object_id,
                    row.guide_object_id,
                    row.detected_object_x_mm,
                    row.detected_object_y_mm,
                    row.guide_object_x_mm,
                    row.guide_object_y_mm,
                    row.detected_object_x_pix,
                    row.detected_object_y_pix,
                    row.guide_object_x_pix,
                    row.guide_object_y_pix,
                    row.agc_camera_id,
                    row.matched,
                )
                for row in self.identified_objects.itertuples(index=False)
            ],
            dtype=[
                ("detected_object_id", np.int16),
                ("guide_object_id", np.int16),
                ("detected_object_x_mm", np.float32),
                ("detected_object_y_mm", np.float32),
                ("guide_object_x_mm", np.float32),
                ("guide_object_y_mm", np.float32),
                ("detected_object_x_pix", np.float32),
                ("detected_object_y_pix", np.float32),
                ("guide_object_x_pix", np.float32),
                ("guide_object_y_pix", np.float32),
                ("camera_id", np.int16),
                ("matched", np.uint8),
            ],
        )
        save_files["identified_objects"] = ident_npy

        full_files = []
        for obj_name, obj in save_files.items():
            fn = os.path.join(base_dir, f"{obj_name}.npy")
            logger.info(f"Saving {obj_name} to {fn}")
            np.save(fn, obj)
            full_files.append(fn)

        return full_files

    def __str__(self) -> str:
        # Counts for arrays (avoid dumping arrays themselves)
        n_guide = 0 if self.guide_objects is None else int(len(self.guide_objects))
        n_detected = (
            0 if self.detected_objects is None else int(len(self.detected_objects))
        )
        n_matched = (
            0
            if self.identified_objects is None
            else int(len(self.identified_objects.query("matched == 1")))
        )

        parts = [
            f"Frame: frame_id={self.frame_id} visit_id={self.visit_id} design_id={self.design_id}",
            f"Field: RA={self.ra:.6f} deg, Dec={self.dec:.6f} deg, PA={self.inst_pa:.3f} deg",
            (
                "Offsets: "
                f"dRA={self.ra_offset:.3f} arcsec "
                f"dDec={self.dec_offset:.3f} arcsec "
                f"dINR={self.inr_offset:.3f} arcsec "
                f"dScale={self.scale_offset:.6f} "
                f"dAlt={self.dalt:.3f} arcsec "
                f"dAz={self.daz:.3f} arcsec "
                f"dx={self.dx:.3f} pix dy={self.dy:.3f} pix"
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
        agc_exposure_info = (
            query_agc_exposure(frame_id, as_dataframe=True)
        )
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
            tel_status_info = (
                query_tel_status(visit_id, sequence_id, as_dataframe=True)
            )
            db_inr = float(tel_status_info.insrot) if pd.notna(tel_status_info.insrot) else db_inr
            db_adc = float(tel_status_info.adc_pa) if pd.notna(tel_status_info.adc_pa) else db_adc
            db_m2_pos3 = float(tel_status_info.m2_pos3) if pd.notna(tel_status_info.m2_pos3) else db_m2_pos3
            db_taken_at = tel_status_info.created_at

        # Use database values for any missing parameters
        taken_at = taken_at or db_taken_at
        inr = inr or db_inr
        adc = adc or db_adc
        m2_pos3 = m2_pos3 or db_m2_pos3

    logger.info(f"tel_status: {taken_at=},{inr=},{adc=},{m2_pos3=}")
    return taken_at, inr, adc, m2_pos3


def get_guide_objects(
    *,
    design_id: int,
    visit0: int | None = None,
    design_path: str | None = None,
    is_guide: bool = False,
    **kwargs,
) -> GuideCatalog:
    """Get the guide objects for a given frame or from other sources.

    Parameters
    ----------
    design_id : int
        The PFS design ID.
    visit0 : int
        The PFS visit ID to look up proper pfs_config_agc data. If no entry exists for
        the given visit_id or if `visit0=None`, the latest entry for the design_id
        is used with appropriate coordinate adjustments.
    design_path : str, optional
        The path to the PFS design file. If None, guide objects are fetched from opdb if design_id is provided.
    is_guide : bool, optional
        If True, guide objects are filtered to only include high quality stars.
    **kwargs : dict
        Additional keyword arguments:

        - ra, dec, inst_pa : Field center coordinates and position angle
        - taken_at, inr, adc, m2_pos3 : Telescope status parameters

    Returns
    -------
    GuideCatalog
        A dataclass containing:

        - guide_objects : np.ndarray
            The guide objects.
        - ra : float
            The right ascension of the field.
        - dec : float
            The declination of the field.
        - inr : float
            The instrument rotator angle.
        - inst_pa : float
            The instrument position angle.
        - m2_pos3 : float
            The M2 position 3 value.
        - adc : float
            The ADC setting.
        - taken_at : datetime
            The time the frame was taken.
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
    ra = ra or field_design.field_ra
    dec = dec or field_design.field_dec
    inst_pa = inst_pa or field_design.field_inst_pa

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
    """Update the RA/Dec and focal-plane positions for the pfsDesign guide objects.

    Adjusts guide star positions for proper motion and parallax based on the observation
    time and field center coordinates.

    See `pfs.utils.pfsConfigUtils.tweakTargetPosition` for reference.

    Parameters
    ----------
    guide_objects : pd.DataFrame
        DataFrame containing guide star data with columns:
        - guide_star_ra: Right ascension in degrees
        - guide_star_dec: Declination in degrees
        - pm_ra: Proper motion in RA (mas/yr)
        - pm_dec: Proper motion in Dec (mas/yr)
        - parallax: Parallax in mas
        - agc_camera_id: Camera ID number
    field_ra : float
        Right ascension of the field center in degrees
    field_dec : float
        Declination of the field center in degrees
    field_pa : float
        Position angle of the field in degrees
    obstime : datetime | str
        Observation time as datetime object with timezone or 'now'

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with updated columns:
        - guide_star_ra: Position-adjusted RA
        - guide_star_dec: Position-adjusted Dec
        - agc_final_x_pix: X pixel coordinate on detector
        - agc_final_y_pix: Y pixel coordinate on detector
    """

    guide_objects = guide_objects.copy()

    logger.info(f"Updating guide object positions from pfsDesign for telescope pointing ({field_ra=},{field_dec=},{field_pa=}) at {obstime=})")

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


def get_detected_objects(
    frame_id: int, filter_flags: int | None = BAD_DETECTION_FLAGS
) -> pd.DataFrame:
    """Get the detected objects from opdb.agc_data.

    Parameters:
    -----------
        frame_id: int
            The frame id of the frame.
        filter_flags: SourceDetectionFlag | int
            Flag to filter out detected objects. Defaults to BAD_DETECTION_FLAGS.

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
    logger.debug(f"Detected objects: {len(detected_objects)}")

    if filter_flags:
        logger.info(f"Filtering detected objects with bad flags: {filter_flags=}")
        detected_objects = detected_objects[(detected_objects['flags'] & int(filter_flags)) == 0]
        logger.debug(f"Detected objects after filtering bad flags: {len(detected_objects)=}")

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
    db: DB | None = None,
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
        db (DB | None): The database to use. Defaults to None, which uses `pfs.utils.database.opdb.OpDB`.
    """
    db = db or OpDB()
    try:
        params = dict(
            agc_exposure_id=frame_id,
            guide_ra=float(ra),
            guide_dec=float(dec),
            guide_pa=float(pa),
            guide_delta_ra=float(delta_ra),
            guide_delta_dec=float(delta_dec),
            guide_delta_insrot=float(delta_insrot),
            guide_delta_scale=float(delta_scale),
            guide_delta_az=float(delta_az) if delta_az is not None else None,
            guide_delta_el=float(delta_el) if delta_el is not None else None,
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
        db.insert_kw("agc_guide_offset", **params)
    except Exception as e:
        logger.warning(f"Failed to write agc_guide_offsets: {e}")


def write_agc_match(
    *,
    design_id: int,
    frame_id: int,
    guide_objects: pd.DataFrame,
    detected_objects: pd.DataFrame,
    identified_objects: pd.DataFrame,
    db: DB | None = None,
) -> int | None:
    """Insert AG identified objects into opdb.agc_match.

    Parameters:
    -----------
    design_id (int): The PFS design ID.
    frame_id (int): The exposure ID for the AGC frame.
    guide_objects (pd.DataFrame): Dictionary or structured array containing guide star data.
    detected_objects (pd.DataFrame): Dictionary or structured array containing detected object data.
    identified_objects (pd.DataFrame): An iterable of tuples, where each tuple contains
                                   indices and coordinate data for a matched object.
                                   Expected format: (detected_idx, guide_idx,
                                   center_x, center_y, nominal_x, nominal_y, ...)
    db (DB | None): The database to use. Defaults to None, which uses `pfs.utils.database.opdb.OpDB`.

    Returns:
    --------
    int | None
        The number of identified objects inserted or None if no matches.
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
        objects = GaiaDB().query_array(query)
    except Exception as e:
        raise RuntimeError(f"Failed to query Gaia DR3 sources: {e:r}")

    # Return results as an astropy Table
    return Table(rows=objects, names=columns, units=units)


def query_db(
    sql: str,
    params: dict | list | None = None,
    as_dataframe: bool = True,
    single_as_series: bool = False,
    db: DB | None = None,
) -> pd.DataFrame | pd.Series | np.ndarray | None:
    """Helper method to return rows from the sql query either.

    Parameters
    ----------
    sql : str
        The sql query to execute.
    params : dict | list | None
        The parameters to pass to the sql query.
    as_dataframe : bool
        Whether to return a pandas dataframe from the sql query. If only one row
        is returned, return a pandas Series. Defaults to True.
    single_as_series : bool
        Whether to return a pandas series from the sql query. If only one row
        is returned, return a pandas Series. Defaults to False.
    db : DB | None
        The database to use. Defaults to None, which uses `pfs.utils.database.opdb.OpDB`.

    Returns
    -------
    pd.DataFrame | pd.Series | np.ndarray | None
        The results from the query.
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
WHERE agc_exposure_id = :agc_exposure_id
ORDER BY agc_camera_id, spot_id
"""
    params = {"agc_exposure_id": agc_exposure_id}
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
        The PFS visit ID.
    as_dataframe : bool, optional
        Whether to return a pandas DataFrame, by default True.
    **kwargs
        Additional keyword arguments passed to `query_db`.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The result of the query as a pandas DataFrame or numpy array.
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


def filter_guide_objects(
    guide_objects, is_guide=False, flag_column="flags"
) -> pd.DataFrame:
    """Apply filtering to the guide objects based on their flags.

    This function always filters galaxies and binary stars.  If `is_acquisition` is
    `True`, this will look for high quality GAIA stars by including stars with the
    following flags:

    - `GAIA`
    - `PHOTO_SIG`
    - `ASTROMETRIC`
    - `PMRA`
    - `PMDEC`
    - `PARA`

    Parameters
    ----------
    guide_objects : pd.DataFrame
        DataFrame containing guide object data including a column for flags.
    is_guide : bool, optional
        If True, filter the objects to only include high quality GAIA stars, default False.
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
    try:
        galaxy_idx = (
            guide_objects_df[flag_column].values & AutoGuiderStarMask.GALAXY
        ) != 0
        guide_objects_df.loc[galaxy_idx, "filtered_by"] = (
            AutoGuiderStarMask.GALAXY.value
        )
        logger.info(f"Filtered {galaxy_idx.sum()} galaxies from results.")

        if is_guide:
            filters_for_inclusion = [
                AutoGuiderStarMask.NON_BINARY,
                AutoGuiderStarMask.GAIA,
                AutoGuiderStarMask.PHOTO_SIG,
                AutoGuiderStarMask.ASTROMETRIC,
                AutoGuiderStarMask.PMRA,
                AutoGuiderStarMask.PMDEC,
                AutoGuiderStarMask.PARA,
            ]

            # Go through the filters and mark which stars would be flagged as NOT meeting the mask requirement.
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
