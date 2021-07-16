import numpy
from opdb import opDB as opdb
import kawanomoto


def focus(frame_id, verbose=False, logger=None):

    logger and logger.info('frame_id={}'.format(frame_id))

    detected_objects = opdb.query_agc_data(frame_id)

    return _focus(detected_objects, verbose=verbose, logger=logger)


def _focus(detected_objects, verbose=False, logger=None):

    #logger and logger.info('detected_objects={}'.format(detected_objects))

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = numpy.sqrt(numpy.square((x2 - y2) / 2) + numpy.square(xy))
        a = numpy.sqrt(p + q)
        b = numpy.sqrt(p - q)
        return a, b

    _detected_objects = numpy.array([
        (
            x[0],  # camera_id
            x[1],  # spot_id
            0,  # centroid_x (unused)
            0,  # centroid_y (unused)
            0,  # flux (unused)
            *semi_axes(x[5], x[6], x[7]),  # semi-major and semi-minor axes
            x[-1]  # flags
        ) for x in detected_objects
    ])

    pfs = kawanomoto.FieldAcquisitionAndFocusing.PFS()
    #dzs = pfs.Focus(_detected_objects, verbose=verbose)
    dzs = pfs.Focus(_detected_objects)
    logger and logger.info('dzs={}'.format(dzs))
    dz = numpy.nanmedian(dzs)
    logger and logger.info('dz={}'.format(dz))

    if verbose:

        return dz

    return dz


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='focus')
    dz = focus(args.frame_id, verbose=args.verbose, logger=logger)
    print('dz={}'.format(dz))
    if args.verbose:
        pass
