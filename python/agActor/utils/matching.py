"""Derive ``SourceMatchingFlags`` for the detections matched by ``RADECInRShiftA``.

The matcher pairs every filtered detection with its nearest catalog star and then
rejects outliers from the astrometric fit. It keeps the reasons a pair is not a
good match (no star nearby, residual too large, another detection nearer to the
same star) in separate arrays; this module folds them into one flag per
detection, the value written to ``agc_match.flags`` in the opdb.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pfs.utils.datamodel.ag import SourceMatchingFlags


def match_flags(
    close: ArrayLike,
    inlier: ArrayLike,
    nearest_distance: ArrayLike,
    catalog_index: ArrayLike,
) -> NDArray[np.int_]:
    """Return a ``SourceMatchingFlags`` value for each detection.

    Parameters
    ----------
    close : array of bool, shape (N,)
        The detection's nearest catalog star is within the match radius.
    inlier : array of bool, shape (N,)
        The detection's post-fit residual is within the rejection threshold.
    nearest_distance : array of float, shape (N,)
        Distance from the detection to its nearest catalog star [mm].
    catalog_index : array of int, shape (N,)
        Index of that nearest catalog star.

    Returns
    -------
    flags : array of int, shape (N,)
        ``NO_MATCH`` when not close. Otherwise ``BAD_RESIDUAL`` when not an
        inlier, ``UNUSED_MULTI_MATCH`` when another close detection is nearer
        to the same catalog star (both bits when both apply), and
        ``GOOD_MATCH`` when neither applies: matched within tolerance, the
        sole claimant of its star, residual within the threshold.
        ``GOOD_MATCH`` is 1, the value readers already keep.

        The flag says how well a detection matches its star, not whether it
        entered the fit: a detection from a disabled camera can be
        ``GOOD_MATCH`` without ever entering it (``camera_enabled``, column
        10 of the match result, says which; INSTRM-3057 records it in opdb),
        and a duplicate flagged ``UNUSED_MULTI_MATCH`` may have entered it.
    """
    close = np.asarray(close, dtype=bool)
    inlier = np.asarray(inlier, dtype=bool)
    nearest_distance = np.asarray(nearest_distance, dtype=float)
    catalog_index = np.asarray(catalog_index, dtype=int)
    n = len(close)
    if not (len(inlier) == len(nearest_distance) == len(catalog_index) == n):
        raise ValueError("close, inlier, nearest_distance and catalog_index must have the same length")

    # Among the close detections sharing a catalog star, the nearest one (the lowest
    # index on a tie) is the match; the others are unused duplicates.
    order = np.lexsort((np.arange(n), nearest_distance, catalog_index))
    order = order[close[order]]
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = catalog_index[order[1:]] != catalog_index[order[:-1]]
    unused = np.zeros(n, dtype=bool)
    unused[order[~is_first]] = True

    flags = np.zeros(n, dtype=int)
    flags[~close] = SourceMatchingFlags.NO_MATCH
    flags[close & ~inlier] |= SourceMatchingFlags.BAD_RESIDUAL
    flags[close & unused] |= SourceMatchingFlags.UNUSED_MULTI_MATCH
    flags[close & (flags == 0)] = SourceMatchingFlags.GOOD_MATCH
    return flags
