from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy._typing import ArrayLike

# import _gen2_gaia_annulus as gaia
from agActor.kawanomoto import Subaru_POPT2_PFS, Subaru_POPT2_PFS_AG
from agActor.utils import _KEYMAP, filter_kwargs, get_guide_objects, map_kwargs, parse_kwargs, to_altaz
from agActor import coordinates
from agActor.opdb import opDB as opdb
from agActor.utils import get_offset_info


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


def get_tel_status(*, frame_id, logger=None, **kwargs):

    taken_at = kwargs.get('taken_at')
    inr = kwargs.get('inr')
    adc = kwargs.get('adc')
    m2_pos3 = kwargs.get('m2_pos3')
    if any(x is None for x in (taken_at, inr, adc, m2_pos3)):
        visit_id, _, _taken_at, _, _, _inr, _adc, _, _, _, _m2_pos3 = opdb.query_agc_exposure(frame_id)
        if (sequence_id := kwargs.get('sequence_id')) is not None:
            logger and logger.info(f"visit_id={visit_id},sequence_id={sequence_id}")
            # use visit_id from agc_exposure table
            _, _, _inr, _adc, _m2_pos3, _, _, _, _, _taken_at = opdb.query_tel_status(visit_id, sequence_id)
        if taken_at is None:
            taken_at = _taken_at
        if inr is None:
            inr = _inr
        if adc is None:
            adc = _adc
        if m2_pos3 is None:
            m2_pos3 = _m2_pos3
    logger and logger.info(f"taken_at={taken_at},inr={inr},adc={adc},m2_pos3={m2_pos3}")
    return taken_at, inr, adc, m2_pos3


def acquire_field(*,
                  frame_id: int,
                  obswl: float = 0.62,
                  altazimuth: bool = False,
                  logger=None,
                  **kwargs
                  ) -> OffsetInfo:

    def log_info(msg, level='info'):
        if logger is not None:
            getattr(logger, level)(msg)

    log_info(f'Getting detected objects for {frame_id=}')
    detected_objects = opdb.query_agc_data(frame_id)
    log_info(f'Retrieved {len(detected_objects)} detected objects from the database for {frame_id=}.')

    if len(detected_objects) == 0:
        raise RuntimeError("No detected objects found, can't compute offset")

    parse_kwargs(kwargs)

    log_info('Getting tel_status information for {frame_id=}')
    taken_at, inr, adc, m2_pos3 = get_tel_status(frame_id=frame_id, logger=logger, **kwargs)

    # Remove the taken_at provided by the kwargs
    kwargs.pop('taken_at', None)

    log_info('Getting guide objects for acquire_field')
    design_id = kwargs.pop('design_id')
    design_path = kwargs.pop('design_path')
    guide_objects, ra, dec, inst_pa = get_guide_objects(
        design_id, design_path, taken_at, obswl, logger=logger, **kwargs
    )

    # Convert to arcsec.
    if 'dra' in kwargs:
        ra += kwargs.get('dra') / 3600
    if 'ddec' in kwargs:
        dec += kwargs.get('ddec') / 3600
    if 'dpa' in kwargs:
        inst_pa += kwargs.get('dpa') / 3600
    log_info(f'{ra},{dec},{inst_pa}')

    _kwargs = filter_kwargs(kwargs)
    log_info(f"{_kwargs=}")

    offset_info = get_offset_info(
        guide_objects=guide_objects,
        detected_objects=detected_objects,
        ra=ra,
        dec=dec,
        taken_at=taken_at,
        adc=adc,
        inst_pa=inst_pa,
        m2_pos3=m2_pos3,
        obswl=obswl,
        altazimuth=altazimuth,
        logger=logger,
        **_kwargs
    )
    offset_info.ra = ra
    offset_info.dec = dec
    offset_info.inst_pa = inst_pa

    return offset_info


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
    flag_values = guide_objects.flag.to_numpy()

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
        detected_objects[:, 2],
        detected_objects[:, 3],
        detected_objects[:, 4],  # This is not used and I'm guessing incorrect index.
        detected_objects[:, 7],
        v_0,
        v_1
    )

    return ra_offset, dec_offset, inr_offset, scale_offset, mr, md, flag_values


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    parser.add_argument('--center', default=None, help='field center coordinates ra, dec[, pa] (deg)')
    parser.add_argument('--offset', default=None, help='field offset coordinates dra, ddec[, dpa[, dinr]] (arcsec)')
    parser.add_argument('--dinr', type=float, default=None, help='instrument rotator offset, east of north (arcsec)')
    parser.add_argument('--magnitude', type=float, default=None, help='magnitude limit')
    parser.add_argument('--fit-dinr', action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help='')
    parser.add_argument('--fit-dscale', action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help='')
    parser.add_argument('--max-ellipticity', type=float, default=argparse.SUPPRESS, help='')
    parser.add_argument('--max-size', type=float, default=argparse.SUPPRESS, help='')
    parser.add_argument('--min-size', type=float, default=argparse.SUPPRESS, help='')
    parser.add_argument('--max-residual', type=float, default=argparse.SUPPRESS, help='')
    args, _ = parser.parse_known_args()

    kwargs = {}
    if any(x is not None for x in (args.design_id, args.design_path)):
        kwargs['design'] = args.design_id, args.design_path
    if args.center is not None:
        kwargs['center'] = tuple([float(x) for x in args.center.split(',')])
    if args.offset is not None:
        kwargs['offset'] = tuple([float(x) for x in args.offset.split(',')])
    if args.dinr is not None:
        kwargs['dinr'] = args.dinr
    if args.magnitude is not None:
        kwargs['magnitude'] = args.magnitude
    kwargs |= {key: getattr(args, key) for key in _KEYMAP if key in args}
    print('kwargs={}'.format(kwargs))

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    ra, dec, inst_pa, dra, ddec, dinr, dscale, *values = acquire_field(
        frame_id=args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, logger=logger, **kwargs
    )
    print('ra={},dec={},inst_pa={},dra={},ddec={},dinr={},dscale={}'.format(ra, dec, inst_pa, dra, ddec, dinr, dscale))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    # print('guide_objects={}'.format(guide_objects))
    # print('detected_objects={}'.format(detected_objects))
    # print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
