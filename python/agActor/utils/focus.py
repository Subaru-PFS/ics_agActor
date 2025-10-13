import numpy as np

from agActor.coordinates.FieldAcquisitionAndFocusing import calculate_focus_errors
from agActor.utils.logging import log_message
from agActor.utils.math import semi_axes
from agActor.utils.data import query_agc_data

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
    detected_objects = query_agc_data(frame_id)
    _kwargs = _filter_kwargs(kwargs)
    log_message(logger, f"In focus with _kwargs={_kwargs}")
    return _focus(detected_objects, logger=logger, **_kwargs)


def _focus(detected_objects, logger=None, **kwargs):
    _detected_objects = np.array(
        [
            (
                row["agc_camera_id"] + 1,  # camera_id (1-6)
                row["spot_id"],  # spot_id
                0,  # centroid_x (unused)
                0,  # centroid_y (unused)
                0,  # flux (unused)
                *semi_axes(
                    row["central_image_moment_11_pix"],
                    row["central_image_moment_20_pix"],
                    row["central_image_moment_02_pix"],
                ),  # semi-major and semi-minor axes
                row["flags"],  # flags
            )
            for idx, row in detected_objects.query('flags < 2').iterrows()
        ]
    )
    _kwargs = _map_kwargs(kwargs)
    log_message(logger, f"In _focus with _kwargs={_kwargs}")
    dzs = calculate_focus_errors(_detected_objects, **_kwargs)
    log_message(logger, f"dzs={dzs}")
    dz = np.nanmedian(dzs)
    log_message(logger, f"dz={dz}")
    return dz, dzs
