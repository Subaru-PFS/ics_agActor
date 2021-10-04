import astrometry
import field_acquisition
from opdb import opDB as opdb


class Field:

    design_id = 0
    ra = 0.0
    dec = 0.0
    guide_objects = []


def set_design(design_id, logger=None):

    logger and logger.info('design_id={}'.format(design_id))

    _, ra, dec, *_ = opdb.query_pfs_design(design_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    Field.design_id = design_id
    Field.ra = ra
    Field.dec = dec
    Field.guide_objects = []


def set_catalog(frame_id=None, obswl=0.62, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if frame_id is not None:

        # create guide object catalog from frame

        ra = Field.ra
        dec = Field.dec

        _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
        logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

        detected_objects = opdb.query_agc_data(frame_id)

        guide_objects = astrometry.measure(detected_objects, ra, dec, taken_at, inr, adc, m2_pos3=m2_pos3, obswl=obswl, logger=logger)

    else:

        # use guide object catalog from operational database

        design_id = Field.design_id

        guide_objects = opdb.query_pfs_design_agc(design_id)

    Field.guide_objects = guide_objects


def autoguide(frame_id, guide_objects=None, ra=None, dec=None, obswl=0.62, verbose=False, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if guide_objects is None:
        guide_objects = Field.guide_objects
    if ra is None:
        ra = Field.ra
    if dec is None:
        dec = Field.dec
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

    detected_objects = opdb.query_agc_data(frame_id)

    _, _, dinr, dalt, daz, *values = field_acquisition._acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=True, verbose=verbose, logger=logger)

    return (dalt, daz, dinr, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=int, required=True, help='design identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_design(args.design_id, logger=logger)
    set_catalog(args.ref_frame_id, obswl=args.obswl, logger=logger)
    dalt, daz, dinr, *values = autoguide(args.frame_id, obswl=args.obswl, verbose=args.verbose, logger=logger)
    print('dalt={},daz={},dinr={}'.format(dalt, daz, dinr))
    if args.verbose:
        guide_objects, detected_objects, identified_objects, *_ = values
        print(guide_objects)
        print(detected_objects)
        print(identified_objects)
