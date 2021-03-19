import astrometry
import field_acquisition
import opdb


class Field:

    tile_id = 0
    ra = 0.0
    dec = 0.0
    guide_objects = []


def set_tile(tile_id, logger=None):

    logger and logger.info('tile_id={}'.format(tile_id))

    ra, dec, _ = opdb.query_tile(tile_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    Field.tile_id = tile_id
    Field.ra = ra
    Field.dec = dec
    Field.guide_objects = []


def set_catalog(frame_id=None, obswl=0.77, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if frame_id is not None:

        # create guide object catalog from frame

        ra = Field.ra
        dec = Field.dec

        _, _, taken_at, _, _, inr, adc, inside_temperature, _, _, _, _, _ = opdb.query_agc_exposure(frame_id)
        logger and logger.info('taken_at={},inr={},adc={},inside_temperature={}'.format(taken_at, inr, adc, inside_temperature))

        detected_objects = opdb.query_agc_data(frame_id)

        guide_objects = astrometry.measure(detected_objects, ra, dec, taken_at, inr, adc, obswl=obswl, inside_temperature=inside_temperature, logger=logger)

    else:

        # use guide object catalog from operational database

        tile_id = Field.tile_id

        guide_objects = opdb.query_guide_star(tile_id)

    Field.guide_objects = guide_objects


def autoguide(frame_id, guide_objects=None, ra=None, dec=None, obswl=0.77, verbose=False, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    if guide_objects is None:
        guide_objects = Field.guide_objects
    if ra is None:
        ra = Field.ra
    if dec is None:
        dec = Field.dec
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    _, _, taken_at, _, _, inr, adc, inside_temperature, _, _, _, _, _ = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},inside_temperature={}'.format(taken_at, inr, adc, inside_temperature))

    detected_objects = opdb.query_agc_data(frame_id)

    _, _, dinr, dalt, daz, *extra = field_acquisition._acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, obswl=obswl, inside_temperature=inside_temperature, altazimuth=True, verbose=verbose, logger=logger)

    return (dalt, daz, dinr, *extra)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--tile-id', type=int, required=True, help='tile identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--ref-frame-id', type=int, default=None, help='reference frame identifier')
    parser.add_argument('--obswl', type=float, default=0.77, help='wavelength of observation (um)')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='autoguide')
    set_tile(args.tile_id, logger=logger)
    set_catalog(args.ref_frame_id, obswl=args.obswl, logger=logger)
    dalt, daz, dinr, *extra = autoguide(args.frame_id, obswl=args.obswl, verbose=args.verbose, logger=logger)
    print('dalt={},daz={},dinr={}'.format(dalt, daz, dinr))
    if args.verbose:
        guide_objects, detected_objects, identified_objects = extra
        print(guide_objects)
        print(detected_objects)
        print(identified_objects)
