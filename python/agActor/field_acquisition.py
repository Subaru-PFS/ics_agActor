import numpy
import coordinates
import opdb
import to_altaz
import kawanomoto


def acquire_field(tile_id, frame_id, obswl=0.62, altazimuth=False, verbose=False, logger=None):

    ra, dec, _ = opdb.query_tile(tile_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    guide_objects = opdb.query_guide_star(tile_id)

    _, _, taken_at, _, _, inr, adc, _, _, _, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

    detected_objects = opdb.query_agc_data(frame_id)

    return _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=altazimuth, verbose=verbose, logger=logger)


def _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=5.5, obswl=0.62, altazimuth=False, verbose=False, logger=None):

    def semi_axes(mu11, mu20, mu02):

        a = mu20 + mu02
        b = numpy.sqrt(4 * numpy.square(mu11) + numpy.square(mu20 - mu02))

        return numpy.sqrt(2 * (a + b)), numpy.sqrt(2 * (a - b))

    _guide_objects = numpy.array([(15 * x[1], x[2], x[3]) for x in guide_objects])

    _detected_objects = numpy.array([
        (
            x[0],
            x[1],
            *coordinates.det2dp(int(x[0]) - 1, x[3], x[4]),
            x[10],
            *semi_axes(x[5] / x[2], x[6] / x[2], x[7] / x[2]),
            x[-1]
        ) for x in detected_objects
    ])

    pfs = kawanomoto.FieldAcquisition.PFS()
    dra, ddec, dinr, *extra = pfs.FA(_guide_objects, _detected_objects, 15 * ra, dec, taken_at, adc, inr, m2_pos3, obswl, verbose=verbose)
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))

    if verbose:

        v, f, min_dist_index_f, obj_x, obj_y, cat_x, cat_y = extra
        index_v, = numpy.where(v)
        index_f, = numpy.where(f)
        identified_objects = [
            (
                k,  # index of detected object
                int(x[0]),  # index of identified guide object
                float(x[1]), float(x[2]),  # detector plane coordinates of detected object
                float(x[3]), float(x[4]),  # detector plane coordinates of identified guide object
                *coordinates.dp2det(detected_objects[k][0] - 1, float(x[3]), float(x[4]))  # detector coordinates of identified guide object
            ) for k, x in ((int(index_v[int(index_f[i])]), x) for i, x in enumerate(zip(min_dist_index_f, obj_x, obj_y, cat_x, cat_y)))
        ]
        extra = guide_objects, detected_objects, identified_objects

    if altazimuth:

        _, _, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
        logger and logger.info('dalt={},daz={}'.format(dalt, daz))

        return (dra, ddec, dinr, dalt, daz, *extra)

    return (dra, ddec, dinr, *extra)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--tile-id', type=int, required=True, help='tile identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    if args.altazimuth:
        dra, ddec, dinr, dalt, daz, *extra = acquire_field(args.tile_id, args.frame_id, obswl=args.obswl, altazimuth=True, verbose=args.verbose, logger=logger)
        print('dra={},ddec={},dinr={},dalt={},daz={}'.format(dra, ddec, dinr, dalt, daz))
    else:
        dra, ddec, dinr, *extra = acquire_field(args.tile_id, args.frame_id, obswl=args.obswl, verbose=args.verbose, logger=logger)
        print('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
    if args.verbose:
        guide_objects, detected_objects, identified_objects = extra
        print(guide_objects)
        print(detected_objects)
        print(identified_objects)
