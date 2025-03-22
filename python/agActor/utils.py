# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntFlag
from logging import Logger
from numbers import Number

import numpy as np
import pandas as pd
from astropy import units
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from numpy._typing import ArrayLike

from agActor import _gen2_gaia as gaia, coordinates, subaru
from agActor.kawanomoto import Subaru_POPT2_PFS, Subaru_POPT2_PFS_AG
from agActor.opdb import opDB as opdb
from agActor.pfs_design import pfsDesign as pfs_design

_KEYMAP = {
    'fit_dinr': ('inrflag', int),
    'fit_dscale': ('scaleflag', int),
    'max_ellipticity': ('maxellip', float),
    'max_size': ('maxsize', float),
    'min_size': ('minsize', float),
    'max_residual': ('maxresid', float),
    'magnitude': ('magnitude', float),
}

FILENAMES = {
    'guide_objects': '/dev/shm/guide_objects.npy',
    'detected_objects': '/dev/shm/detected_objects.npy',
    'identified_objects': '/dev/shm/identified_objects.npy'
}


# TODO Use a shared version from pfs.datamodel.
class AutoGuiderStarMask(IntFlag):
    """
    Represents a bitmask for guide star properties.

    Attributes:
        GAIA: Gaia DR3 catalog.
        HSC: HSC PDR3 catalog.
        PMRA: Proper motion RA is measured.
        PMRA_SIG: Proper motion RA measurement is significant (SNR>5).
        PMDEC: Proper motion Dec is measured.
        PMDEC_SIG: Proper motion Dec measurement is significant (SNR>5).
        PARA: Parallax is measured.
        PARA_SIG: Parallax measurement is significant (SNR>5).
        ASTROMETRIC: Astrometric excess noise is small (astrometric_excess_noise<1.0).
        ASTROMETRIC_SIG: Astrometric excess noise is significant (astrometric_excess_noise_sig>2.0).
        NON_BINARY: Not a binary system (RUWE<1.4).
        PHOTO_SIG: Photometric measurement is significant (SNR>5).
        GALAXY: Is a galaxy candidate.
    """
    GAIA = 0x00001
    HSC = 0x00002
    PMRA = 0x00004
    PMRA_SIG = 0x00008
    PMDEC = 0x00010
    PMDEC_SIG = 0x00020
    PARA = 0x00040
    PARA_SIG = 0x00080
    ASTROMETRIC = 0x00100
    ASTROMETRIC_SIG = 0x00200
    NON_BINARY = 0x00400
    PHOTO_SIG = 0x00800
    GALAXY = 0x01000


@dataclass
class OffsetInfo:
    ra: float | None = None
    dec: float | None = None
    inst_pa: float | None = None
    dra: float | None = None
    ddec: float | None = None
    dinr: float | None = None
    dscale: float | None = None
    dalt: float | None = None
    daz: float | None = None
    dx: float | None = None
    dy: float | None = None
    spot_size: float | None = None
    peak_intensity: float | None = None
    flux: float | None = None
    guide_objects: ArrayLike | None = None
    detected_objects: ArrayLike | None = None
    identified_objects: ArrayLike | None = None


def parse_kwargs(kwargs: dict) -> None:

    if (center := kwargs.pop('center', None)) is not None:
        ra, dec, *optional = center
        kwargs.setdefault('ra', ra)
        kwargs.setdefault('dec', dec)
        kwargs.setdefault('inst_pa', optional[0] if len(optional) > 0 else 0)

    if (offset := kwargs.pop('offset', None)) is not None:
        dra, ddec, *optional = offset
        kwargs.setdefault('dra', dra)
        kwargs.setdefault('ddec', ddec)
        kwargs.setdefault('dpa', optional[0] if len(optional) > 0 else kwargs.get('dinr', 0))
        kwargs.setdefault('dinr', optional[1] if len(optional) > 1 else optional[0] if len(optional) > 0 else 0)

    if (design := kwargs.pop('design', None)) is not None:
        design_id, design_path = design
        kwargs.setdefault('design_id', design_id)
        kwargs.setdefault('design_path', design_path)

    if (status_id := kwargs.pop('status_id', None)) is not None:
        visit_id, sequence_id = status_id
        kwargs.setdefault('visit_id', visit_id)
        kwargs.setdefault('sequence_id', sequence_id)

    if (tel_status := kwargs.pop('tel_status', None)) is not None:
        _, _, inr, adc, m2_pos3, _, _, _, taken_at = tel_status
        kwargs.setdefault('taken_at', taken_at)
        kwargs.setdefault('inr', inr)
        kwargs.setdefault('adc', adc)
        kwargs.setdefault('m2_pos3', m2_pos3)


def filter_kwargs(kwargs):

    return {k: v for k, v in kwargs.items() if k in _KEYMAP}


def map_kwargs(kwargs: dict) -> dict:

    return {_KEYMAP[k][0]: _KEYMAP[k][1](v) for k, v in kwargs.items() if k in _KEYMAP}


def to_altaz(
    ra,
    dec,
    obstime=None,
    temperature=0,
    relative_humidity=0,
    pressure=620,
    obswl=0.62,
    dra=0,
    ddec=0
):

    ra = Angle(ra, unit=units.deg)
    dec = Angle(dec, unit=units.deg)
    obstime = Time(obstime.astimezone(tz=timezone.utc)) if isinstance(obstime, datetime) else Time(
        obstime, format='unix'
    ) if isinstance(
        obstime, Number
    ) else Time(obstime) if obstime is not None else Time.now()

    temperature *= units.deg_C
    relative_humidity /= 100
    pressure *= units.hPa
    obswl *= units.micron

    dra *= units.arcsec
    ddec *= units.arcsec

    frame = AltAz(
        obstime=obstime, location=subaru.location, temperature=temperature, relative_humidity=relative_humidity,
        pressure=pressure, obswl=obswl
    )

    icrs = SkyCoord(ra=[ra, ra + dra], dec=[dec, dec + ddec], frame='icrs')
    altaz = icrs.transform_to(frame)

    alt = altaz[0].alt.to(units.deg).value
    az = altaz[0].az.to(units.deg).value

    dalt = (altaz[1].alt - altaz[0].alt).to(units.arcsec).value
    daz = (altaz[1].az - altaz[0].az).to(units.arcsec).value

    return alt, az, dalt, daz


def get_guide_objects(
    design_id: int | None = None,
    design_path: str | None = None,
    taken_at: datetime | Number | str | None = None,
    obswl: float = 0.62,
    apply_filters: bool = True,
    logger=None,
    **kwargs
) -> tuple[pd.DataFrame, float, float, float]:
    def log_info(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    if design_path is not None:
        log_info('Getting guide_objects from the design file.')
        guide_objects, ra, dec, inst_pa = pfs_design(
            design_id,
            design_path,
            logger=logger
        ).get_guide_objects(
            taken_at=taken_at
        )
    elif design_id is not None:
        log_info('Getting guide_objects from the operational database.')
        _, ra, dec, inst_pa, *_ = opdb.query_pfs_design(design_id)
        guide_objects = opdb.query_pfs_design_agc(design_id)
    else:
        taken_at = kwargs.get('taken_at')
        adc = kwargs.get('adc')
        m2_pos3 = kwargs.get('m2_pos3', 6.0)
        log_info(f"{taken_at=},{adc=},{m2_pos3=}")

        ra = kwargs.get('ra')
        dec = kwargs.get('dec')
        inst_pa = kwargs.get('inst_pa')
        log_info(f'ra={ra},dec={dec},inst_pa={inst_pa}')

        log_info('Generating guide_objects on-the-fly (from gaia database).')
        guide_objects, *_ = gaia.get_objects(
            ra=ra, dec=dec, obstime=taken_at, inst_pa=inst_pa, adc=adc, m2pos3=m2_pos3, obswl=obswl
        )

    # Use Table to convert, which handles big-endian and little-endian issues.
    guide_objects = Table(guide_objects).to_pandas()
    log_info(f'Got {len(guide_objects)} guide objects.')

    guide_objects.columns = ['objId', 'epoch', 'ra', 'dec', 'pmRa', 'pmDec', 'parallax', 'magnitude', 'passband',
                             'color', 'agId', 'agX', 'agY', 'flag']

    # Add a column to indicate which flat was used for filtering.
    guide_objects['filtered_by'] = 0

    if apply_filters:
        filters_for_inclusion = [AutoGuiderStarMask.GAIA,
                                 AutoGuiderStarMask.NON_BINARY,
                                 AutoGuiderStarMask.ASTROMETRIC,
                                 AutoGuiderStarMask.PMRA_SIG,
                                 AutoGuiderStarMask.PMDEC_SIG,
                                 AutoGuiderStarMask.PARA_SIG,
                                 AutoGuiderStarMask.PHOTO_SIG]

        # Filter the guide objects to only include the ones that are not flagged as galaxies.
        log_info('Filtering guide objects to remove galaxies.')
        galaxy_idx = (guide_objects.flag.values & AutoGuiderStarMask.GALAXY) != 0
        guide_objects.loc[galaxy_idx, 'filtered_by'] = AutoGuiderStarMask.GALAXY.value
        log_info(f'Filtering by {AutoGuiderStarMask.GALAXY.name}, removes {galaxy_idx.sum()} guide objects.')

        # The initial coarse guide uses all the stars and the fine guide uses only the GAIA stars.
        coarse = kwargs.get('coarse', False)
        if coarse is False:
            # Go through the filters and mark which stars would be flagged as NOT meeting the mask requirement.
            for f in filters_for_inclusion:
                not_filtered = guide_objects.filtered_by == 0
                include_filter = (guide_objects.flag.values & f) == 0
                to_be_filtered = (include_filter & not_filtered) != 0
                guide_objects.loc[to_be_filtered, 'filtered_by'] = f.value
                print(f'Filtering by {f.name}, removes {to_be_filtered.sum()} guide objects.')

    return guide_objects, ra, dec, inst_pa


def get_offset_info(
    guide_objects: pd.DataFrame,
    detected_objects: pd.DataFrame,
    ra: float,
    dec: float,
    taken_at: datetime,
    adc: float,
    inst_pa: float = 0.0,
    m2_pos3: float = 6.0,
    obswl: float = 0.62,
    altazimuth: bool = False,
    logger: Logger | None = None,
    **kwargs
) -> OffsetInfo:
    _kwargs = map_kwargs(kwargs)
    logger and logger.info(f"{_kwargs=}")

    if isinstance(taken_at, datetime):
        taken_at = taken_at.astimezone(tz=timezone.utc)
    elif isinstance(taken_at, Number):
        taken_at = datetime.fromtimestamp(taken_at, tz=timezone.utc)
    else:
        taken_at = taken_at

    detector_plane_coords = detected_objects.apply(
        lambda row: coordinates.det2dp(row.camera_id, row.centroid_x, row.centroid_y), axis=1, result_type='expand'
        )

    sa_moments = detected_objects.apply(
        lambda row: semi_axes(row.central_moment_11, row.central_moment_20, row.central_moment_02),
        axis=1, result_type='expand'
        )

    detected_objects['focal_plane_x_mm'] = detector_plane_coords[0]
    detected_objects['focal_plane_y_mm'] = detector_plane_coords[1]
    detected_objects['semi_axes_a'] = sa_moments[0]
    detected_objects['semi_axes_b'] = sa_moments[1]

    good_guide_objects = guide_objects.query('filtered_by == 0')

    ra_offset, dec_offset, inr_offset, scale_offset, mr, md, detected_objects_flags = calculate_offset(
        good_guide_objects,
        detected_objects,
        ra,
        dec,
        taken_at,
        adc,
        inst_pa,
        m2_pos3,
        obswl,
        _kwargs
    )

    mr_df = pd.DataFrame(mr)
    mr_df['camera_id'] = detected_objects.iloc[np.where(detected_objects_flags)].camera_id.values
    # Filter bad values TODO fix column names.
    mr_df = mr_df[mr_df[8] == 1]
    guide_idx = mr_df[9].astype(int).values
    fp_coords = mr_df.apply(lambda row: coordinates.dp2det(row.camera_id, row[2], row[3]), axis=1, result_type='expand')

    identified_objects = pd.DataFrame({
        'detected_object_idx': mr_df.index.values,
        'guide_obj_idx': guide_idx,
        'detected_object_x': mr_df[0],
        'detected_object_y': mr_df[1],
        'guide_object_x': mr_df[2],
        'guide_object_y': mr_df[3],
        'guide_object_xdet': fp_coords[0],
        'guide_object_ydet': fp_coords[1],
    })

    ra_offset *= 3600
    dec_offset *= 3600
    inr_offset *= 3600
    logger and logger.info(f"{ra_offset=},{dec_offset=},{inr_offset=},{scale_offset=}")
    dalt = None
    daz = None

    if altazimuth:
        alt, az, dalt, daz = to_altaz(ra, dec, taken_at, dra=ra_offset, ddec=dec_offset)
        logger and logger.info(f"{alt=},{az=},{dalt=},{daz=}")

    # guide_objects = np.array(
    #     [(x.iloc[0],
    #       x.iloc[1],
    #       x.iloc[2],
    #       x.iloc[3],
    #       x.iloc[4],
    #       x.iloc[5],
    #       x.iloc[6],
    #       x.iloc[7],
    #       x.iloc[9],
    #       x.iloc[10],
    #       x.iloc[11],
    #       x.iloc[12],
    #       x.iloc[13]) for idx, x in guide_objects.iterrows()],
    #     dtype=[
    #         ('source_id', np.int64),  # u8 (80) not supported by FITSIO
    #         ('epoch', str),
    #         ('ra', np.float64),
    #         ('dec', np.float64),
    #         ('pmRa', np.float32),
    #         ('pmDec', np.float32),
    #         ('parallax', np.float32),
    #         ('magnitude', np.float32),
    #         ('color', np.float32),
    #         ('camera_id', np.int16),
    #         ('guide_object_xdet', np.float32),
    #         ('guide_object_ydet', np.float32),
    #         ('flags', np.uint16)
    #     ]
    # )

    # detected_objects = np.array(
    #     detected_objects,
    #     dtype=[
    #         ('camera_id', np.int16),
    #         ('spot_id', np.int16),
    #         ('moment_00', np.float32),
    #         ('centroid_x', np.float32),
    #         ('centroid_y', np.float32),
    #         ('central_moment_11', np.float32),
    #         ('central_moment_20', np.float32),
    #         ('central_moment_02', np.float32),
    #         ('peak_x', np.uint16),
    #         ('peak_y', np.uint16),
    #         ('peak', np.uint16),
    #         ('background', np.float32),
    #         ('flags', np.uint8)
    #     ]
    # )

    # index_v = np.where(detected_flags)[0]
    # identified_objects = np.array(
    #     [
    #         (
    #             k,  # index of detected object
    #             int(x[0]),  # index of identified guide object
    #             float(x[1]), float(x[2]),  # detector plane coordinates of detected object
    #             float(x[3]), float(x[4]),  # detector plane coordinates of identified guide object
    #             *coordinates.dp2det(detected_objects[k][0], float(x[3]), float(x[4]))
    #             # detector coordinates of identified guide object
    #         )
    #         for k, x in
    #         ((int(index_v[i]), x) for i, x in enumerate(zip(mr[:, 9], mr[:, 0], mr[:, 1], mr[:, 2], mr[:, 3], mr[:,
    #         8]))
    #          if int(x[5]))
    #     ],
    #     dtype=[
    #         ('detected_object_id', np.int16),
    #         ('guide_object_id', np.int16),
    #         ('detected_object_x', np.float32),
    #         ('detected_object_y', np.float32),
    #         ('guide_object_x', np.float32),
    #         ('guide_object_y', np.float32),
    #         ('guide_object_xdet', np.float32),
    #         ('guide_object_ydet', np.float32)
    #     ]
    # )
    dx = - ra_offset * np.cos(np.deg2rad(dec))  # arcsec
    dy = dec_offset  # arcsec (HSC definition)

    # find "representative" spot size, peak intensity, and flux by "median" of pointing errors
    spot_size = 0  # pix
    peak_intensity = 0  # pix
    flux = 0  # pix

    # squares of pointing errors in detector plane coordinates
    square_pointing_errors = (identified_objects['detected_object_x'] - identified_objects['guide_object_x']) ** 2 + (
        identified_objects['detected_object_y'] - identified_objects['guide_object_y']) ** 2

    num_errors = len(square_pointing_errors) - np.isnan(square_pointing_errors).sum()
    logger and logger.info(f"Number of errors: {num_errors}")
    if num_errors > 0:
        try:
            identified_median_idx = np.argpartition(square_pointing_errors, num_errors // 2)[num_errors // 2]
            detected_median_idx = identified_objects['detected_object_idx'][identified_median_idx]

            a, b = semi_axes(
                detected_objects['central_moment_11'][detected_median_idx],
                detected_objects['central_moment_20'][detected_median_idx],
                detected_objects['central_moment_02'][detected_median_idx]
            )

            spot_size = (a * b) ** 0.5
            peak_intensity = detected_objects['peak'][detected_median_idx]
            flux = detected_objects['moment_00'][detected_median_idx]
        except Exception as e:
            logger and logger.error(f"Error calculating median values: {e}")
            spot_size = 0
            peak_intensity = 0
            flux = 0


    return OffsetInfo(
        dra=ra_offset,
        ddec=dec_offset,
        dinr=inr_offset,
        dscale=scale_offset,
        dalt=dalt,
        daz=daz,
        dx=dx,
        dy=dy,
        spot_size=spot_size,
        peak_intensity=peak_intensity,
        flux=flux,
        guide_objects=guide_objects.to_numpy(),
        detected_objects=detected_objects.to_numpy(),
        identified_objects=identified_objects.to_numpy(),
    )


def semi_axes(xy, x2, y2):
    p = (x2 + y2) / 2
    q = np.sqrt(np.square((x2 - y2) / 2) + np.square(xy))
    a = np.sqrt(p + q)
    b = np.sqrt(p - q)
    return a, b


def calculate_offset(guide_objects: pd.DataFrame, detected_objects, ra, dec, taken_at, adc, inst_pa, m2_pos3, obswl,
                     kwargs
                     ):
    """Calculate the offset of the field.

    This method replaces the functionality of `FAinstpa` so we can remove the filtering.
    """
    subaru = Subaru_POPT2_PFS.Subaru()
    inr0 = subaru.radec2inr(ra, dec, taken_at)
    inr = inr0 + inst_pa

    pfs = Subaru_POPT2_PFS_AG.PFS()

    # RA [2], Dec [3], PM RA [4], PM Dec [5], Parallax [6], Magnitude [7], Flags [-1]
    ra_values = guide_objects.ra.to_numpy()
    dec_values = guide_objects.dec.to_numpy()
    magnitude_values = guide_objects.magnitude.to_numpy()
    flag_values = detected_objects.flag < 2
    filtered_detected_objects = detected_objects[flag_values].copy()

    v_0, v_1 = pfs.makeBasis(
        ra,
        dec,
        ra_values,
        dec_values,
        taken_at,
        adc,
        inr,
        m2_pos3,
        obswl
    )
    v_0 = (np.insert(v_0, 2, magnitude_values, axis=1))
    v_1 = (np.insert(v_1, 2, magnitude_values, axis=1))

    # Get the offsets.
    ra_offset, dec_offset, inr_offset, scale_offset, mr, md = pfs.RADECInRScaleShift(
        filtered_detected_objects.focal_plane_x_mm.values,
        filtered_detected_objects.focal_plane_y_mm.values,
        filtered_detected_objects.camera_id.values,  # UNUSED
        filtered_detected_objects.flag.values,
        v_0,
        v_1
    )

    return ra_offset, dec_offset, inr_offset, scale_offset, mr, md, flag_values
