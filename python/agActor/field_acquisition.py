import numpy
import det2dp
import opdb
import to_altaz
import kawanomoto


def acquire_field(target_id, frame_id, obswl=0.77, altazimuth=False, logger=None):

    ra, dec, _ = opdb.query_target(target_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    guide_objects = opdb.query_guide_star(target_id)

    _, _, taken_at, _, _, inr, adc, inside_temperature, _, _, _, _, _ = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},inside_temperature={}'.format(taken_at, inr, adc, inside_temperature))

    detected_objects = opdb.query_agc_data(frame_id)

    return _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, obswl=obswl, inside_temperature=inside_temperature, altazimuth=altazimuth, logger=logger)


def _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, obswl=0.77, inside_temperature=0, altazimuth=False, logger=None):

    def semi_axes(mu11, mu20, mu02):

        a = mu20 + mu02
        b = numpy.sqrt(4 * numpy.square(mu11) + numpy.square(mu20 - mu02))

        return numpy.sqrt(2 * (a + b)), numpy.sqrt(2 * (a - b))

    _guide_objects = numpy.array([(15 * x[1], x[2], x[3]) for x in guide_objects])

    _detected_objects = numpy.array(
        [
            (
                x[0],
                x[1],
                *det2dp.det2dp(int(x[0]) - 1, x[3], x[4]),
                x[10],
                *semi_axes(x[5] / x[2], x[6] / x[2], x[7] / x[2])
            ) for x in detected_objects
        ]
    )

    pfs = kawanomoto.FieldAcquisition.PFS()
    dra, ddec, dinr = pfs.FA(_guide_objects, _detected_objects, 15 * ra, dec, taken_at, adc, inr, inside_temperature + 273.15, obswl)
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))

    if altazimuth:

        _, _, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
        logger and logger.info('dalt={},daz={}'.format(dalt, daz))

        return dra, ddec, dinr, dalt, daz

    return dra, ddec, dinr


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--target-id', type=int, required=True, help='target identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.77, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    if args.altazimuth:
        dra, ddec, dinr, dalt, daz = acquire_field(args.target_id, args.frame_id, obswl=args.obswl, altazimuth=True, logger=logger)
        print('dra={},ddec={},dinr={},dalt={},daz={}'.format(dra, ddec, dinr, dalt, daz))
    else:
        dra, ddec, dinr = acquire_field(args.target_id, args.frame_id, obswl=args.obswl, logger=logger)
        print('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
