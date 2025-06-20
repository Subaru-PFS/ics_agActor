import numpy

from kawanomoto import FieldAcquisitionAndFocusing
from opdb import opDB as opdb

# mapping of keys and value types between focus.py and FieldAcquisitionAndFocusing.py
_KEYMAP = {
    "max_ellipticity": ("maxellip", float),
    "max_size": ("maxsize", float),
    "min_size": ("minsize", float),
}


def _filter_kwargs(kwargs):

    return {k: v for k, v in kwargs.items() if k in _KEYMAP}


def _map_kwargs(kwargs):

    return {_KEYMAP[k][0]: _KEYMAP[k][1](v) for k, v in kwargs.items() if k in _KEYMAP}


def focus(*, frame_id, logger=None, **kwargs):

    logger and logger.info("frame_id={}".format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id)
    _kwargs = _filter_kwargs(kwargs)
    logger and logger.info("_kwargs={}".format(_kwargs))
    return _focus(detected_objects, logger=logger, **_kwargs)


def _focus(detected_objects, logger=None, **kwargs):

    # logger and logger.info('detected_objects={}'.format(detected_objects))

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
                x[-1],  # flags
            )
            for x in detected_objects
        ]
    )
    _kwargs = _map_kwargs(kwargs)
    logger and logger.info("_kwargs={}".format(_kwargs))
    pfs = FieldAcquisitionAndFocusing.PFS()
    dzs = pfs.Focus(_detected_objects, **_kwargs)
    logger and logger.info("dzs={}".format(dzs))
    dz = numpy.nanmedian(dzs)
    logger and logger.info("dz={}".format(dz))
    return dz, dzs


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-id", type=int, required=True, help="frame identifier")
    parser.add_argument("--max-ellipticity", type=float, default=argparse.SUPPRESS, help="")
    parser.add_argument("--max-size", type=float, default=argparse.SUPPRESS, help="")
    parser.add_argument("--min-size", type=float, default=argparse.SUPPRESS, help="")
    args, _ = parser.parse_known_args()

    kwargs = {key: getattr(args, key) for key in _KEYMAP if key in args}
    print("kwargs={}".format(kwargs))

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name="focus")
    dz, _ = focus(frame_id=args.frame_id, logger=logger, **kwargs)
    print("dz={}".format(dz))
