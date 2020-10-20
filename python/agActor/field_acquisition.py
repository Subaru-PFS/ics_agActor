import numpy
import det2dp
import opdb
import kawanomoto


def acquire_field(target_id, frame_id, logger=None):

    ra, dec, _ = opdb.query_target(target_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))
    guide_objects = opdb.query_guide_star(target_id)
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    _, _, taken_at, _, _, inr, adc = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={}'.format(taken_at, inr, adc))
    detected_objects = opdb.query_agc_data(frame_id)
    #logger and logger.info('detected_objects={}'.format(detected_objects))
    return _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, logger=logger)


def _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, logger=None):

    def semi_axes(mu11, mu20, mu02):

        a = mu20 + mu02
        b = numpy.sqrt(4 * numpy.square(mu11) + numpy.square(mu20 - mu02))
        return numpy.sqrt(2 * (a + b)), numpy.sqrt(2 * (a - b))

    _guide_objects = numpy.array([(15 * x[1], x[2], x[3]) for x in guide_objects])
    #logger and logger.info('_guide_objects={}'.format(_guide_objects))
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
    #logger and logger.info('_detected_objects={}'.format(_detected_objects))
    pfs = kawanomoto.FieldAcquisition.PFS()
    dra, ddec, dinr = pfs.FA(_guide_objects, _detected_objects, 15 * ra, dec, taken_at, adc, inr)
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
    return dra, ddec, dinr


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('target_id', type=int, help='target identifier')
    parser.add_argument('frame_id', type=int, help='frame identifier')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    dra, ddec, dinr = acquire_field(args.target_id, args.frame_id, logger=logger)
    print('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
