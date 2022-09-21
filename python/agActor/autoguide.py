import astrometry
import field_acquisition
import _gen2_gaia as gaia
#import _gen2_gaia_annulus as gaia
from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design


class Field:

    design = None
    center = None
    guide_objects = None


def set_design(*, logger=None, **kwargs):

    field_acquisition._parse_kwargs(kwargs)
    design_id = kwargs.get('design_id')
    design_path = kwargs.get('design_path')
    logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))
    ra = kwargs.get('ra')
    dec = kwargs.get('dec')
    inst_pa = kwargs.get('inst_pa')
    if any(x is None for x in (ra, dec, inst_pa)):
        if any(x is not None for x in (design_id, design_path)):
            if design_path is not None:
                _ra, _dec, _inst_pa = pfs_design(design_id, design_path, logger=logger).center
            else:
                _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
            if ra is None: ra = _ra
            if dec is None: dec = _dec
            if inst_pa is None: inst_pa = _inst_pa
    logger and logger.info('ra={},dec={},inst_pa={}'.format(ra, dec, inst_pa))
    Field.design = design_id, design_path
    Field.center = ra, dec, inst_pa
    Field.guide_objects = []  # delay loading of guide objects


def set_design_agc(*, frame_id=None, obswl=0.62, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    field_acquisition._parse_kwargs(kwargs)
    if frame_id is not None:
        # generate guide objects from frame
        ra, dec, *_ = Field.center
        logger and logger.info('ra={},dec={}'.format(ra, dec))
        taken_at, inr, adc, m2_pos3 = field_acquisition._get_tel_status(frame_id=frame_id, logger=logger, **kwargs)
        detected_objects = opdb.query_agc_data(frame_id)
        #logger and logger.info('detected_objects={}'.format(detected_objects))
        if 'dra' in kwargs: ra += kwargs.get('dra') / 3600
        if 'ddec' in kwargs: dec += kwargs.get('ddec') / 3600
        if 'dinr' in kwargs: inr += kwargs.get('dinr') / 3600
        logger and logger.info('ra={},dec={},inr={}'.format(ra, dec, inr))
        guide_objects = astrometry.measure(detected_objects=detected_objects, ra=ra, dec=dec, obstime=taken_at, inr=inr, adc=adc, m2_pos3=m2_pos3, obswl=obswl, logger=logger)
    else:
        # use guide objects from pfs design file or operational database, or generate on-the-fly
        design_id, design_path = Field.design
        logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))
        if design_path is not None:
            taken_at = kwargs.get('taken_at')
            magnitude = kwargs.get('magnitude', 20.0)
            logger and logger.info('taken_at={},magnitude={}'.format(taken_at, magnitude))
            guide_objects, *_ = pfs_design(design_id, design_path, logger=logger).guide_objects(magnitude=magnitude, obstime=taken_at)
        elif design_id is not None:
            guide_objects = opdb.query_pfs_design_agc(design_id)
        else:
            ra, dec, *_ = Field.center
            logger and logger.info('ra={},dec={}'.format(ra, dec))
            taken_at = kwargs.get('taken_at')
            inr = kwargs.get('inr')
            adc = kwargs.get('adc')
            m2_pos3 = kwargs.get('m2_pos3', 6.0)
            magnitude = kwargs.get('magnitude', 20.0)
            logger and logger.info('taken_at={},inr={},adc={},m2_pos3={},magnitude={}'.format(taken_at, inr, adc, m2_pos3, magnitude))
            if 'dra' in kwargs: ra += kwargs.get('dra') / 3600
            if 'ddec' in kwargs: dec += kwargs.get('ddec') / 3600
            if 'dinr' in kwargs: inr += kwargs.get('dinr') / 3600
            logger and logger.info('ra={},dec={},inr={}'.format(ra, dec, inr))
            guide_objects, *_ = gaia.get_objects(ra=ra, dec=dec, obstime=taken_at, inr=inr, adc=adc, m2pos3=m2_pos3, obswl=obswl, magnitude=magnitude)
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    Field.guide_objects = guide_objects


def autoguide(*, frame_id, obswl=0.62, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    field_acquisition._parse_kwargs(kwargs)
    guide_objects = Field.guide_objects
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    ra, dec, *_ = Field.center
    logger and logger.info('ra={},dec={}'.format(ra, dec))
    taken_at, inr, adc, m2_pos3 = field_acquisition._get_tel_status(frame_id=frame_id, logger=logger, **kwargs)
    detected_objects = opdb.query_agc_data(frame_id)
    #logger and logger.info('detected_objects={}'.format(detected_objects))
    if 'dra' in kwargs: ra += kwargs.get('dra') / 3600
    if 'ddec' in kwargs: dec += kwargs.get('ddec') / 3600
    if 'dinr' in kwargs: inr += kwargs.get('dinr') / 3600
    logger and logger.info('ra={},dec={},inr={}'.format(ra, dec, inr))
    return field_acquisition._acquire_field(guide_objects=guide_objects, detected_objects=detected_objects, ra=ra, dec=dec, taken_at=taken_at, adc=adc, inr=inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=True, logger=logger)  # (dra, ddec, dinr, dalt, daz, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--center', default=None, help='field center coordinates ra, dec[, pa] (deg)')
    parser.add_argument('--offset', default=None, help='field offset coordinates dra, ddec[, dpa[, dinr]] (arcsec)')
    parser.add_argument('--dinr', type=float, default=None, help='instrument rotator offset, east of north (arcsec)')
    parser.add_argument('--magnitude', type=float, default=20.0, help='magnitude limit')
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
    kwargs['magnitude'] = args.magnitude

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_design(logger=logger, **kwargs)
    set_design_agc(frame_id=args.ref_frame_id, obswl=args.obswl, logger=logger, **kwargs)
    dra, ddec, dinr, dalt, daz, *values = autoguide(frame_id=args.frame_id, obswl=args.obswl, logger=logger, **kwargs)
    print('dra={},ddec={},dinr={},dalt={},daz={}'.format(dra, ddec, dinr, dalt, daz))
    guide_objects, detected_objects, identified_objects, *_ = values
    #print(guide_objects)
    #print(detected_objects)
    #print(identified_objects)
