from logging import Logger

import numpy
import pandas as pd
from numpy._typing import ArrayLike

from kawanomoto import FieldAcquisitionAndFocusing
from agActor.opdb import opDB as opdb
from agActor.utils import semi_axes
from agActor.utils import _KEYMAP, filter_kwargs, map_kwargs


def focus(*, frame_id, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id, as_dataframe=True)
    _kwargs = filter_kwargs(kwargs)
    logger and logger.info('_kwargs={}'.format(_kwargs))
    return _focus(detected_objects, logger=logger, **_kwargs)


def _focus(detected_objects: pd.DataFrame, logger: Logger | None = None, **kwargs):
    _detected_objects = numpy.array(
        [
            (
                x['camera_id'] + 1,  # camera_id (1-6)
                x['spot_id'],  # spot_id
                0,  # centroid_x (unused)
                0,  # centroid_y (unused)
                0,  # flux (unused)
                *semi_axes(x['central_moment_11'], x['central_moment_20'], x['central_moment_02']),  # semi-major and semi-minor axes
                x['flag']  # flags
            )
            for idx, x in detected_objects.iterrows()
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
