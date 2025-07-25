"""
Mathematical utility functions for the agActor package.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def semi_axes(
    xy: float | NDArray, x2: float | NDArray, y2: float | NDArray
) -> Tuple[float | NDArray, float | NDArray]:
    """
    Calculate the semi-major and semi-minor axes of an ellipse using eigenvalues.

    Parameters
    ----------
    xy : float or numpy.ndarray
        The cross-term coefficient in the ellipse equation.
    x2 : float or numpy.ndarray
        The x-squared coefficient in the ellipse equation.
    y2 : float or numpy.ndarray
        The y-squared coefficient in the ellipse equation.

    Returns
    -------
    Tuple[float | NDArray, float | NDArray]
        A tuple containing (semi-major axis, semi-minor axis) of the ellipse.
        Returns floats if inputs are floats, or numpy arrays if inputs are arrays.
    """
    p = (x2 + y2) / 2
    q = np.sqrt(np.square((x2 - y2) / 2) + np.square(xy))
    a = np.sqrt(p + q)
    b = np.sqrt(p - q)
    return a, b
