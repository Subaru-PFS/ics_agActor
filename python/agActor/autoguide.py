from logging import Logger

import astrometry
import field_acquisition

from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design
from agActor.utils import get_offset_info
from agActor.field_acquisition import OffsetInfo
from agActor.utils import filter_kwargs, get_guide_objects, parse_kwargs, _KEYMAP


class Field:

    design = None
    center = None
    guide_objects = None


def set_field(*, frame_id=None, obswl=0.62, logger=None, **kwargs):

    def log_info(msg):
        if logger is not None:
            logger.info(msg)

    parse_kwargs(kwargs)

    design_id, design_path, ra, dec, inst_pa = get_field_info(kwargs, logger)

    Field.design = design_id, design_path
    Field.center = ra, dec, inst_pa
    Field.guide_objects = []  # clear values and delay loading of guide objects

    log_info(f"{frame_id=}")
    taken_at = kwargs.get('taken_at')
    log_info(f"taken_at={taken_at}")

    if frame_id is not None:
        detected_objects = opdb.query_agc_data(frame_id)
        log_info(f'Retrieved {len(detected_objects)} detected objects from the database for {frame_id=}.')

        # generate guide objects from frame.
        taken_at, inr, adc, m2_pos3 = field_acquisition.get_tel_status(frame_id=frame_id, logger=logger, **kwargs)

        # Convert to arcsec
        if 'dra' in kwargs:
            ra += kwargs.get('dra') / 3600
        if 'ddec' in kwargs:
            dec += kwargs.get('ddec') / 3600

        # TODO get the expected XY positions and cam_id in from here.
        log_info('Getting guide_objects from astrometric measurement of the detected objects.')
        guide_objects = astrometry.measure(
            detected_objects=detected_objects,
            ra=ra, dec=dec, obstime=taken_at, inst_pa=inst_pa, adc=adc,
            m2_pos3=m2_pos3, obswl=obswl, logger=logger
        )
    else:
        # use guide objects from pfs design file or operational database, or generate on-the-fly.
        guide_objects, ra, dec, inst_pa = get_guide_objects(
            design_id, design_path, taken_at, obswl, logger=logger, **kwargs
        )

    log_info(f'Setting the guide objects for the field with {len(guide_objects)} objects.')
    Field.guide_objects = guide_objects


def get_field_info(kwargs: dict, logger: Logger | None = None) -> tuple[float, int | None, str | None, float, float]:
    """Get the field center information."""
    design_id = kwargs.get('design_id')
    design_path = kwargs.get('design_path')
    ra = kwargs.get('ra')
    dec = kwargs.get('dec')
    inst_pa = kwargs.get('inst_pa')
    # If we don't have all the center info, try to get from the design file by path or id.
    if any(x is None for x in (ra, dec, inst_pa)):
        if any(x is not None for x in (design_id, design_path)):
            if design_path is not None:
                _ra, _dec, _inst_pa = pfs_design(design_id, design_path, logger=logger).center
            else:
                _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
            ra = ra or _ra
            dec = dec or _dec
            inst_pa = inst_pa or _inst_pa

    logger.info(f"design_id={design_id},design_path={design_path}")
    logger.info(f"ra={ra},dec={dec},inst_pa={inst_pa}")

    return design_id, design_path, ra, dec, inst_pa


def acquire_field(*, frame_id, obswl=0.62, logger=None, **kwargs) -> OffsetInfo:
    def log_info(msg):
        if logger is not None:
            logger.info(msg)

    log_info(f'Getting detected objects for {frame_id=}')
    detected_objects = opdb.query_agc_data(frame_id)
    log_info(f'Retrieved {len(detected_objects)} detected objects from the database for {frame_id=}.')

    parse_kwargs(kwargs)

    log_info('Getting tel_status information for {frame_id=}')
    taken_at, inr, adc, m2_pos3 = field_acquisition.get_tel_status(frame_id=frame_id, logger=logger, **kwargs)

    log_info('Getting guide objects for autoguide')
    guide_objects = Field.guide_objects
    ra, dec, inst_pa = Field.center

    # Convert to arcsec.
    if 'dra' in kwargs:
        ra += kwargs.get('dra') / 3600
    if 'ddec' in kwargs:
        dec += kwargs.get('ddec') / 3600
    if 'dpa' in kwargs:
        inst_pa += kwargs.get('dpa') / 3600
    log_info(f"{ra=},{dec=},{inst_pa=}")

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
        altazimuth=True,
        logger=logger,
        **_kwargs
    )

    offset_info.ra = ra
    offset_info.dec = dec
    offset_info.inst_pa = inst_pa

    return offset_info


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
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
    logger = logging.getLogger(name='autoguide')
    set_field(frame_id=args.ref_frame_id, obswl=args.obswl, logger=logger, **kwargs)
    ra, dec, inst_pa, dra, ddec, dinr, dscale, dalt, daz, *values = acquire_field(
        frame_id=args.frame_id, obswl=args.obswl, logger=logger, **kwargs
    )
    print(
        'ra={},dec={},inst_pa={},dra={},ddec={},dinr={},dscale={},dalt={},daz={}'.format(
            ra, dec, inst_pa, dra, ddec, dinr, dscale, dalt, daz
        )
    )
    guide_objects, detected_objects, identified_objects, *_ = values
    # print(guide_objects)
    # print(detected_objects)
    # print(identified_objects)
