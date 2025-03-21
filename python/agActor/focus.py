from logging import Logger

import numpy
from numpy._typing import ArrayLike

from kawanomoto import FieldAcquisitionAndFocusing
from opdb import opDB as opdb
from python.agActor.field_acquisition import semi_axes
from agActor.utils import _KEYMAP, filter_kwargs, map_kwargs


def focus(*, frame_id, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id)
    _kwargs = filter_kwargs(kwargs)
    logger and logger.info('_kwargs={}'.format(_kwargs))
    return _focus(detected_objects, logger=logger, **_kwargs)


def _focus(detected_objects: ArrayLike, logger: Logger | None = None, **kwargs):
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
    _kwargs = map_kwargs(kwargs)
    logger and logger.info(f"_kwargs={_kwargs}")

    pfs = FieldAcquisitionAndFocusing.PFS()
    dzs = pfs.Focus(_detected_objects, **_kwargs)

    logger and logger.info(f"{dzs=}")
    median_dz = numpy.nanmedian(dzs)
    logger and logger.info(f"{median_dz=}")

    return median_dz, dzs


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--max-ellipticity', type=float, default=argparse.SUPPRESS, help='')
    parser.add_argument('--max-size', type=float, default=argparse.SUPPRESS, help='')
    parser.add_argument('--min-size', type=float, default=argparse.SUPPRESS, help='')
    args, _ = parser.parse_known_args()

    kwargs = {key: getattr(args, key) for key in _KEYMAP if key in args}
    print('kwargs={}'.format(kwargs))

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='focus')
    dz, _ = focus(frame_id=args.frame_id, logger=logger, **kwargs)
    print('dz={}'.format(dz))
