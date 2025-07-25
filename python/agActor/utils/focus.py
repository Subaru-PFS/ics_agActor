import numpy as np

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_focus_errors
from agActor.utils.logging import log_message
from agActor.utils.opdb import opDB as opdb

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

    log_message(logger, f"frame_id={frame_id}")
    detected_objects = opdb.query_agc_data(frame_id)
    _kwargs = _filter_kwargs(kwargs)
    log_message(logger, f"_kwargs={_kwargs}")
    return _focus(detected_objects, logger=logger, **_kwargs)


def _focus(detected_objects, logger=None, **kwargs):

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = np.sqrt(np.square((x2 - y2) / 2) + np.square(xy))
        a = np.sqrt(p + q)
        b = np.sqrt(p - q)
        return a, b

    _detected_objects = np.array(
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
    log_message(logger, f"_kwargs={_kwargs}")
    dzs = calculate_focus_errors(_detected_objects, **_kwargs)
    log_message(logger, f"dzs={dzs}")
    dz = np.nanmedian(dzs)
    log_message(logger, f"dz={dz}")
    return dz, dzs
