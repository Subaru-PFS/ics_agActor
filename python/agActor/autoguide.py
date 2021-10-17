import astrometry
import field_acquisition
from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design


class Field:

    design = None
    center = None
    guide_objects = None


def set_design(design=None, logger=None):

    design_id, design_path = design
    logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))

    if design_path is not None:

        _, ra, dec, pa = pfs_design(design_id, design_path).guide_stars

    else:

        _, ra, dec, pa, *_ = opdb.query_pfs_design(design_id)

    logger and logger.info('ra={},dec={}'.format(ra, dec))

    Field.design = design
    Field.center = ra, dec, pa
    Field.guide_objects = []  # delay loading of guide objects


def set_design_agc(frame_id=None, obswl=0.62, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if frame_id is not None:

        # create guide object table from frame

        ra, dec, _ = Field.center

        _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
        logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

        detected_objects = opdb.query_agc_data(frame_id)

        guide_objects = astrometry.measure(detected_objects, ra, dec, taken_at, inr, adc, m2_pos3=m2_pos3, obswl=obswl, logger=logger)

    else:

        # use guide object table from operational database or pfs design file

        design_id, design_path = Field.design

        if design_path is not None:

            guide_objects, *_ = pfs_design(design_id, design_path).guide_stars

        else:

            guide_objects = opdb.query_pfs_design_agc(design_id)

    Field.guide_objects = guide_objects


def autoguide(frame_id, obswl=0.62, verbose=False, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    guide_objects = Field.guide_objects
    ra, dec, _ = Field.center
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

    detected_objects = opdb.query_agc_data(frame_id)

    _, _, dinr, dalt, daz, *values = field_acquisition._acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=True, verbose=verbose, logger=logger)

    return (dalt, daz, dinr, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    if all(x is None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --design-id, --design-path')

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_design(design=(args.design_id, args.design_path), logger=logger)
    set_design_agc(frame_id=args.ref_frame_id, obswl=args.obswl, logger=logger)
    dalt, daz, dinr, *values = autoguide(frame_id=args.frame_id, obswl=args.obswl, verbose=args.verbose, logger=logger)
    print('dalt={},daz={},dinr={}'.format(dalt, daz, dinr))
    if args.verbose:
        guide_objects, detected_objects, identified_objects, *_ = values
        print(guide_objects)
        print(detected_objects)
        print(identified_objects)
