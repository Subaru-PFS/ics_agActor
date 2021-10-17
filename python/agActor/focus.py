import numpy
from opdb import opDB as opdb
import kawanomoto


def focus(frame_id, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id)
    return _focus(detected_objects, logger=logger)


def _focus(detected_objects, logger=None):

    #logger and logger.info('detected_objects={}'.format(detected_objects))

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = numpy.sqrt(numpy.square((x2 - y2) / 2) + numpy.square(xy))
        a = numpy.sqrt(p + q)
        b = numpy.sqrt(p - q)
        return a, b

    _detected_objects = numpy.array(
        [
            (
                x[0] + 1,  # camera_id (1-6)
                x[1],  # spot_id
                0,  # centroid_x (unused)
                0,  # centroid_y (unused)
                0,  # flux (unused)
                *semi_axes(x[5], x[6], x[7]),  # semi-major and semi-minor axes
                x[-1]  # flags
            )
            for x in detected_objects
        ]
    )
    pfs = kawanomoto.FieldAcquisitionAndFocusing.PFS()
    dzs = pfs.Focus(_detected_objects)
    logger and logger.info('dzs={}'.format(dzs))
    dz = numpy.nanmedian(dzs)
    logger and logger.info('dz={}'.format(dz))
    return dz


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='focus')
    dz = focus(args.frame_id, logger=logger)
    print('dz={}'.format(dz))
