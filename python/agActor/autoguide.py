import astrometry
import field_acquisition
import opdb
import to_altaz


class Field:

    target_id = 0
    ra = 0.0
    dec = 0.0
    guide_objects = []


def set_target(target_id, logger=None):

    logger and logger.info('target_id={}'.format(target_id))

    ra, dec, _ = opdb.query_target(target_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    Field.target_id = target_id
    Field.ra = ra
    Field.dec = dec
    Field.guide_objects = []


def set_catalog(frame_id=None, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if frame_id is not None:

        # create guide object catalog from frame

        ra = Field.ra
        dec = Field.dec

        _, _, taken_at, _, _, inr, adc = opdb.query_agc_exposure(frame_id)
        logger and logger.info('taken_at={},inr={},adc={}'.format(taken_at, inr, adc))

        detected_objects = opdb.query_agc_data(frame_id)

        guide_objects = astrometry.measure(detected_objects, ra, dec, taken_at, inr, adc, logger=logger)

    else:

        # use guide object catalog from operational database

        target_id = Field.target_id

        guide_objects = opdb.query_guide_star(target_id)

    Field.guide_objects = guide_objects


def autoguide(frame_id, guide_objects=None, ra=None, dec=None, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if guide_objects is None:
        guide_objects = Field.guide_objects
    if ra is None:
        ra = Field.ra
    if dec is None:
        dec = Field.dec
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    _, _, taken_at, _, _, inr, adc = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={}'.format(taken_at, inr, adc))

    detected_objects = opdb.query_agc_data(frame_id)

    return _autoguide(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, logger=logger)


def _autoguide(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, logger=None):

    dra, ddec, dinr = field_acquisition._acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, logger=logger)
    #logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))

    _, _, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
    logger and logger.info('dalt={},daz={}'.format(dalt, daz))

    return dalt, daz, dinr


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--target-id', type=int, required=True, help='target identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_target(args.target_id, logger=logger)
    set_catalog(args.ref_frame_id, logger=logger)
    dalt, daz, dinr = autoguide(args.frame_id, logger=logger)
    print('dalt={},daz={},dinr={}'.format(dalt, daz, dinr))
