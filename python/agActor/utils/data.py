"""AG actor data-model: dataclasses, constants, and flags.

This module contains only the core data types used across the AG actor:

- :data:`BAD_DETECTION_FLAGS` — detection-quality bitmask constant
  (re-exported from :mod:`agActor.utils.queries` to avoid a circular import)
- :class:`GuideOffsetFlag` — integer flag for guide-offset validity
  (canonical definition is in :mod:`agActor.utils.queries`)
- :class:`GuideCatalog` — dataclass holding the guide-star catalog result
- :class:`GuideOffsets` — dataclass holding the per-frame guiding results

Functions that interact with the database live in
:mod:`agActor.utils.queries`.  Guide-catalog helpers
(``get_guide_objects``, ``filter_guide_objects``, ``tweak_target_position``)
live in :mod:`agActor.utils.guide_catalog`.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# BAD_DETECTION_FLAGS and GuideOffsetFlag are defined in queries.py (their
# primary user) and re-exported here for backward compatibility.
# Importing them before the dataclasses avoids a forward-reference issue.
from agActor.utils.queries import BAD_DETECTION_FLAGS, GuideOffsetFlag  # noqa: E402


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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


__all__ = [
    # Constants
    "BAD_DETECTION_FLAGS",
    # Dataclasses / flags
    "GuideCatalog",
    "GuideOffsets",
    "GuideOffsetFlag",
]
