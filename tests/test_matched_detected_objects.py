"""Focus uses the detections matched to guide stars."""

import pandas as pd

from agActor.utils.data import GuideOffsets
from pfs.utils.datamodel.ag import SourceMatchingFlags as M


def _guide_offsets(identified: pd.DataFrame) -> GuideOffsets:
    detected = pd.DataFrame(
        {"agc_camera_id": [0, 0, 1, 1], "spot_id": [1, 2, 1, 2], "flags": [0, 0, 1, 0]},
        index=[10, 11, 12, 13],
    )
    return GuideOffsets(
        ra=0.0, dec=0.0, inst_pa=0.0, ra_offset=0.0, dec_offset=0.0, inr_offset=0.0, scale_offset=0.0,
        dalt=None, daz=None, guide_objects=pd.DataFrame(),
        detected_objects=detected, identified_objects=identified,
        dx=0.0, dy=0.0, size=0.0, peak=0.0, flux=0.0,
    )


def test_matched_detected_objects_keeps_only_good_matches():
    identified = pd.DataFrame(
        {
            "detected_object_id": [10, 11, 12, 13],
            "match_flags": [int(M.GOOD_MATCH), int(M.NO_MATCH), int(M.GOOD_MATCH), int(M.UNUSED_MULTI_MATCH)],
        }
    )
    identified["matched"] = (identified.match_flags == int(M.GOOD_MATCH)).astype(int)
    go = _guide_offsets(identified)

    matched = go.matched_detected_objects
    assert list(matched.index) == [10, 12]
    assert list(matched.columns) == list(go.detected_objects.columns)
    # Negative control: an unmatched spot, flagged or not, never reaches the focus estimate,
    # and the selection is stricter than the detection flags alone.
    assert 11 not in matched.index and 13 not in matched.index
    assert len(matched) < (go.detected_objects["flags"] <= 1).sum()


def test_matched_detected_objects_empty_when_nothing_matched():
    identified = pd.DataFrame(
        {"detected_object_id": [10, 11], "match_flags": [int(M.NO_MATCH)] * 2, "matched": [0, 0]}
    )
    assert _guide_offsets(identified).matched_detected_objects.empty
