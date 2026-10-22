"""Tests for the SourceMatchingFlags written to agc_match.flags (INSTRM-3038).

The frame in ``FRAME`` is shaped like the ones in the run28/run29 AG logs: about a
dozen filtered detections, nearly every frame with one spot whose nearest guide
star is several mm away, a few residual outliers, and (invisibly, until now) the
occasional pair of detections claiming the same star.
"""

import numpy as np
import pandas as pd
import pytest
from agActor.utils.matching import match_flags
from pfs.utils.datamodel.ag import SourceMatchingFlags as F

# One row per detection: catalog index, distance to it [mm], close?, inlier?
FRAME = pd.DataFrame(
    [
        (0, 0.02, True, True),  # good
        (1, 0.03, True, True),  # good
        (2, 0.04, True, False),  # close but rejected by the residual cut
        (3, 8.10, False, False),  # nearest star 8 mm away: never matched
        (4, 0.01, True, True),  # good; claims star 4 ...
        (4, 0.30, True, True),  # ... as does this one, farther away: unused duplicate
        (5, 0.01, True, False),  # nearest to star 5 but rejected: it still keeps the star ...
        (5, 0.02, True, True),  # ... so this inlier is an unused duplicate, not a good match
        (6, 0.50, True, False),  # duplicate of star 6, farther, and an outlier: both bits
        (6, 0.02, True, True),  # keeps star 6
        (7, np.nan, False, False),  # no finite distance at all
    ],
    columns=["catalog_index", "nearest_distance", "close", "inlier"],
)

EXPECTED = [
    F.GOOD_MATCH,
    F.GOOD_MATCH,
    F.BAD_RESIDUAL,
    F.NO_MATCH,
    F.GOOD_MATCH,
    F.UNUSED_MULTI_MATCH,
    F.BAD_RESIDUAL,
    F.UNUSED_MULTI_MATCH,
    F.BAD_RESIDUAL | F.UNUSED_MULTI_MATCH,
    F.GOOD_MATCH,
    F.NO_MATCH,
]


def _flags(frame=FRAME):
    return match_flags(frame.close, frame.inlier, frame.nearest_distance, frame.catalog_index)


def test_frame_flags():
    flags = _flags()
    assert flags.tolist() == [int(f) for f in EXPECTED]
    assert flags.dtype.kind == "i"


def test_good_match_is_one_and_readers_keep_only_fit_inliers():
    """drp_qa's selectValidMatches keeps ``flags == 1``; that must be the fit's inliers alone."""
    assert int(F.GOOD_MATCH) == 1
    flags = _flags()
    drove_fit = (FRAME.close & FRAME.inlier).to_numpy()
    duplicate = flags & F.UNUSED_MULTI_MATCH != 0
    assert np.array_equal(flags == 1, drove_fit & ~duplicate)
    # Negative control: the old 0/1 convention cannot tell a bad residual from no match,
    # and ``matched`` alone would have kept the unused duplicate.
    assert not np.array_equal(flags == 1, drove_fit)
    assert flags[3] != flags[2] and flags[3] != 0 and flags[2] != 0


def test_every_bad_row_says_why():
    flags = _flags()
    bad = flags != F.GOOD_MATCH
    assert bad.any()
    assert np.all(flags[bad] != F.NONE)
    assert np.all((flags[bad] & F.GOOD_MATCH) == 0)


def test_nearest_duplicate_wins_whatever_the_row_order():
    frame = FRAME.iloc[::-1].reset_index(drop=True)
    assert _flags(frame).tolist() == [int(f) for f in EXPECTED[::-1]]


def test_tie_goes_to_the_first_row():
    frame = pd.DataFrame(
        {"catalog_index": [9, 9], "nearest_distance": [0.1, 0.1], "close": [True, True], "inlier": [True, True]}
    )
    assert _flags(frame).tolist() == [F.GOOD_MATCH, F.UNUSED_MULTI_MATCH]


def test_far_duplicate_is_not_a_duplicate():
    """A detection beyond the match radius is NO_MATCH even if it shares a nearest star."""
    frame = pd.DataFrame(
        {"catalog_index": [9, 9], "nearest_distance": [0.1, 5.0], "close": [True, False], "inlier": [True, False]}
    )
    assert _flags(frame).tolist() == [F.GOOD_MATCH, F.NO_MATCH]


def test_empty_frame():
    assert match_flags([], [], [], []).tolist() == []


def test_length_mismatch():
    with pytest.raises(ValueError):
        match_flags([True], [True, False], [0.1], [0])
