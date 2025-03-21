# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
from datetime import datetime, timezone
from enum import IntFlag
from numbers import Number

import numpy as np
import pandas as pd
from agActor import _gen2_gaia as gaia
from agActor.opdb import opDB as opdb
from agActor.pfs_design import pfsDesign as pfs_design
from astropy import units
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.table import Table
from astropy.time import Time
from agActor import subaru

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
    logger=None,
    **kwargs
) -> tuple[pd.DataFrame, float, float, float]:
    def log_info(msg):
        if logger is not None:
            logger.info(msg)

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

    # Filter the guide objects to only include the ones that are not flagged as galaxies.
    log_info('Filtering guide objects to remove galaxies.')
    galaxy_idx = (guide_objects.flag & np.array(AutoGuiderStarMask.GALAXY)).values.astype(bool)
    guide_objects = guide_objects[~galaxy_idx]
    log_info(f'Got {len(guide_objects)} guide objects after filtering galaxies.')

    # Filter the guide objects to only include the ones that are not flagged as binaries.
    log_info('Filtering guide objects to remove binaries.')
    binary_idx = (guide_objects.flag & np.array(AutoGuiderStarMask.NON_BINARY)).values.astype(bool)
    guide_objects = guide_objects[~binary_idx]
    log_info(f'Got {len(guide_objects)} guide objects after filtering binaries.')

    # The initial coarse guide uses all the stars and the fine guide uses only the GAIA stars.
    initial = kwargs.get('initial', False)
    if initial is False:
        log_info('Filtering guide objects to only GAIA stars.')
        gaia_idx = (guide_objects.flag & np.array(AutoGuiderStarMask.GAIA)).values.astype(bool)
        guide_objects = guide_objects[gaia_idx]
        log_info(f'Got {len(guide_objects)} guide objects after filtering.')

    return guide_objects, ra, dec, inst_pa
