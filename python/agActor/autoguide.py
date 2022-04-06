import astrometry
import field_acquisition
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
                _, _ra, _dec, _inst_pa = pfs_design(design_id, design_path).guide_stars
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
        taken_at = kwargs.get('taken_at')
        inr = kwargs.get('inr')
        adc = kwargs.get('adc')
        m2_pos3 = kwargs.get('m2_pos3')
        if any(x is None for x in (taken_at, inr, adc, m2_pos3)):
            visit_id, _, _taken_at, _, _, _inr, _adc, _, _, _, _m2_pos3 = opdb.query_agc_exposure(frame_id)
            if (sequence_id := kwargs.get('sequence_id')) is not None:
                # use visit_id from agc_exposure table
                _, _, _inr, _adc, _m2_pos3, _, _, _, _, _taken_at = opdb.query_tel_status(visit_id, sequence_id)
            if taken_at is None: taken_at = _taken_at
            if inr is None: inr = _inr
            if adc is None: adc = _adc
            if m2_pos3 is None: m2_pos3 = _m2_pos3
        logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))
        detected_objects = opdb.query_agc_data(frame_id)
        #logger and logger.info('detected_objects={}'.format(detected_objects))
        guide_objects = astrometry.measure(detected_objects=detected_objects, ra=ra, dec=dec, taken_at=taken_at, inr=inr, adc=adc, m2_pos3=m2_pos3, obswl=obswl, logger=logger)
    else:
        # use guide objects from pfs design file or operational database, or generate on-the-fly
        design_id, design_path = Field.design
        logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))
        if design_path is not None:
            guide_objects, *_ = pfs_design(design_id, design_path).guide_stars
        elif design_id is not None:
            guide_objects = opdb.query_pfs_design_agc(design_id)
        else:
            ra, dec, *_ = Field.center
            logger and logger.info('ra={},dec={}'.format(ra, dec))
            guide_objects, *_ = gaia.get_objects(ra=ra, dec=dec, obstime=taken_at, inr=inr, adc=adc, m2pos3=m2_pos3, obswl=obswl)
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    Field.guide_objects = guide_objects


def autoguide(*, frame_id, obswl=0.62, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    field_acquisition._parse_kwargs(kwargs)
    guide_objects = Field.guide_objects
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    ra, dec, *_ = Field.center
    logger and logger.info('ra={},dec={}'.format(ra, dec))
    taken_at = kwargs.get('taken_at')
    inr = kwargs.get('inr')
    adc = kwargs.get('adc')
    m2_pos3 = kwargs.get('m2_pos3')
    if None in (taken_at, inr, adc, m2_pos3):
        visit_id, _, _taken_at, _, _, _inr, _adc, _, _, _, _m2_pos3 = opdb.query_agc_exposure(frame_id)
        if (sequence_id := kwargs.get('sequence_id')) is not None:
            # use visit_id from agc_exposure table
            _, _, _inr, _adc, _m2_pos3, _, _, _, _, _taken_at = opdb.query_tel_status(visit_id, sequence_id)
        if taken_at is None: taken_at = _taken_at
        if inr is None: inr = _inr
        if adc is None: adc = _adc
        if m2_pos3 is None: m2_pos3 = _m2_pos3
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))
    detected_objects = opdb.query_agc_data(frame_id)
    #logger and logger.info('detected_objects={}'.format(detected_objects))
    _, _, dinr, dalt, daz, *values = field_acquisition._acquire_field(guide_objects=guide_objects, detected_objects=detected_objects, ra=ra, dec=dec, taken_at=taken_at, adc=adc, inr=inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=True, logger=logger)
    return (dalt, daz, dinr, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    args, _ = parser.parse_known_args()

    if all(x is None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --design-id, --design-path')

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_design(design=(args.design_id, args.design_path), logger=logger)
    set_design_agc(frame_id=args.ref_frame_id, obswl=args.obswl, logger=logger)
    dalt, daz, dinr, *values = autoguide(frame_id=args.frame_id, obswl=args.obswl, logger=logger)
    print('dalt={},daz={},dinr={}'.format(dalt, daz, dinr))
    guide_objects, detected_objects, identified_objects, *_ = values
    #print(guide_objects)
    #print(detected_objects)
    #print(identified_objects)
