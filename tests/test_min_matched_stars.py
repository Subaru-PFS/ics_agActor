"""An offset is used only when enough guide stars matched (INSTRM-2800)."""

import pandas as pd
import pytest
from pfs.utils.datamodel.ag import SourceMatchingFlags

from agActor.autoguide import classify_offset, count_fit_stars, too_few_stars_message
from agActor.utils.data import GuideOffsetFlag, GuideOffsets

GOOD = int(SourceMatchingFlags.GOOD_MATCH)
NO_MATCH = int(SourceMatchingFlags.NO_MATCH)
BAD_RESIDUAL = int(SourceMatchingFlags.BAD_RESIDUAL)
MULTI = int(SourceMatchingFlags.UNUSED_MULTI_MATCH)


def identified(match_flags, camera_enabled=None):
    """identified_objects as field_acquisition.get_guide_offsets builds them: matched is GOOD_MATCH alone."""
    df = pd.DataFrame({"match_flags": match_flags})
    df["matched"] = (df.match_flags == GOOD).astype(int)
    # The fit's inlier column also counts a duplicate claim on a star (UNUSED_MULTI_MATCH).
    df["inlier"] = df.match_flags.isin([GOOD, MULTI]).astype(float)
    if camera_enabled is not None:
        df["camera_enabled"] = [float(c) for c in camera_enabled]
    return df


def test_counts_good_matches_only():
    # Run 30's failure: ~25 detections paired with the wrong stars, all rejected.
    ident = identified([BAD_RESIDUAL] * 20 + [NO_MATCH] * 5 + [MULTI] * 2 + [GOOD])
    assert count_fit_stars(ident) == 1
    # Negative control: counting the detections or the fit's inliers would have passed the frame.
    assert len(ident) >= 3
    assert int(ident.inlier.sum()) >= 3


def test_disabled_cameras_do_not_count():
    ident = identified([GOOD] * 5, camera_enabled=[1, 1, 0, 0, 0])
    assert count_fit_stars(ident) == 2
    # Negative control: ignoring camera_enabled counts the predictions from disabled cameras.
    assert int(ident.matched.sum()) == 5


def test_without_camera_enabled_column_every_camera_counts():
    assert count_fit_stars(identified([GOOD, GOOD, GOOD, NO_MATCH])) == 3


@pytest.mark.parametrize("in_range", [True, False])
def test_too_few_stars_is_rejected_whatever_the_size(in_range):
    flags, status = classify_offset(2, 3, in_range)
    assert flags == GuideOffsetFlag.INVALID_OFFSET | GuideOffsetFlag.TOO_FEW_STARS
    assert status == "TOO_FEW_STARS"


def test_minimum_is_inclusive():
    assert classify_offset(3, 3) == (GuideOffsetFlag.OK, "OK")
    # Negative control: one fewer is rejected.
    assert classify_offset(2, 3)[0] != GuideOffsetFlag.OK


def test_out_of_range_with_enough_stars():
    assert classify_offset(10, 3, in_range=False) == (GuideOffsetFlag.INVALID_OFFSET, "INVALID_OFFSET")


def test_too_few_stars_keeps_the_invalid_bit():
    # drp_qa's run report counts `mask & 1` as rejected; a TOO_FEW_STARS offset must be counted too.
    flags, _ = classify_offset(0, 3)
    assert int(flags) == 3
    assert int(flags) & int(GuideOffsetFlag.INVALID_OFFSET)
    # Negative control: TOO_FEW_STARS alone would read as accepted there.
    assert not int(GuideOffsetFlag.TOO_FEW_STARS) & int(GuideOffsetFlag.INVALID_OFFSET)


def test_message_counts_the_guide_stars_kept():
    guide_objects = pd.DataFrame({"filtered_by": [0, 0, 0, 0, 8, 8]})
    offsets = GuideOffsets(
        ra=0.0,
        dec=0.0,
        inst_pa=0.0,
        ra_offset=0.0,
        dec_offset=0.0,
        inr_offset=0.0,
        scale_offset=0.0,
        dalt=None,
        daz=None,
        guide_objects=guide_objects,
        detected_objects=pd.DataFrame(index=range(32)),
        identified_objects=identified([NO_MATCH] * 32),
        dx=0.0,
        dy=0.0,
        size=0.0,
        peak=0.0,
        flux=0.0,
    )
    message = too_few_stars_message(offsets, 1, 3)
    assert message == (
        "Too few guide stars matched to use the offset: 1 matched (need 3), 32 spots detected, 4 guide stars"
    )
