"""write_agc_match writes SourceMatchingFlags to agc_match.flags (INSTRM-3038)."""

import pandas as pd
import pytest
from agActor.utils.data import write_agc_match
from pfs.utils.datamodel.ag import SourceMatchingFlags as F


class _CapturingDB:
    """Stands in for pfs.utils.database.db.DB: records the frame it is asked to insert."""

    def __init__(self):
        self.inserted = []

    def insert_dataframe(self, *, df, table):
        self.inserted.append((table, df.copy()))
        return len(df)


@pytest.fixture
def frame():
    detected = pd.DataFrame({"agc_camera_id": [0, 0, 3, 3], "spot_id": [0, 1, 0, 1]})
    guide = pd.DataFrame({"source_id": [1001, 1002, 1003, 1004]})
    identified = pd.DataFrame(
        {
            "detected_object_id": [0, 1, 2, 3],
            "guide_object_id": [0, 1, 1, 3],
            "detected_object_x_mm": [1.0, 2.0, 3.0, 4.0],
            "detected_object_y_mm": [-1.0, -2.0, -3.0, -4.0],
            "guide_object_x_mm": [1.1, 2.1, 3.1, 4.1],
            "guide_object_y_mm": [-1.1, -2.1, -3.1, -4.1],
            "matched": [1, 1, 0, 0],
            "camera_enabled": [1.0, 1.0, 1.0, 1.0],
            "match_flags": [
                int(F.GOOD_MATCH),
                int(F.GOOD_MATCH),
                int(F.UNUSED_MULTI_MATCH),
                int(F.NO_MATCH),
            ],
        }
    )
    return detected, guide, identified


def test_flags_column_carries_match_flags(frame):
    detected, guide, identified = frame
    db = _CapturingDB()
    n = write_agc_match(
        design_id=0x1234, frame_id=42, guide_objects=guide, detected_objects=detected,
        identified_objects=identified, db=db,
    )
    assert n == 4
    (table, df), = db.inserted
    assert table == "agc_match"
    assert df["flags"].tolist() == [1, 1, int(F.UNUSED_MULTI_MATCH), int(F.NO_MATCH)]
    assert df["guide_star_id"].tolist() == [1001, 1002, 1002, 1004]
    assert df["agc_exposure_id"].eq(42).all() and df["pfs_design_id"].eq(0x1234).all()
    # Negative control: the old convention, int(matched), would have written 0 for both
    # bad rows and could not have said why either is bad.
    assert df["flags"].tolist() != identified["matched"].tolist()
    assert (df["flags"] != 0).all()


def test_no_rows_writes_nothing(frame):
    detected, guide, identified = frame
    db = _CapturingDB()
    assert write_agc_match(
        design_id=1, frame_id=1, guide_objects=guide, detected_objects=detected,
        identified_objects=identified.iloc[:0], db=db,
    ) is None
    assert db.inserted == []
