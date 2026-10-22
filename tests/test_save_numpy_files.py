"""GuideOffsets.save_numpy_files saves match_flags, not matched, in identified_objects.npy (INSTRM-3058).

The table vgwActor loads from the ag ``data`` keyword keeps its 12 columns, so vgwActor's
fixed units list still fits; the last column is now the SourceMatchingFlags bitmask.
"""

import numpy as np
import pandas as pd
import pytest
from agActor.utils.data import GuideOffsets
from pfs.utils.datamodel.ag import SourceMatchingFlags as F

IDENT_COLUMNS = (
    "detected_object_id",
    "guide_object_id",
    "detected_object_x_mm",
    "detected_object_y_mm",
    "guide_object_x_mm",
    "guide_object_y_mm",
    "detected_object_x_pix",
    "detected_object_y_pix",
    "guide_object_x_pix",
    "guide_object_y_pix",
    "camera_id",
    "match_flags",
)


@pytest.fixture
def offsets():
    flags = [int(F.GOOD_MATCH), int(F.UNUSED_MULTI_MATCH), int(F.BAD_RESIDUAL | F.UNUSED_MULTI_MATCH)]
    identified = pd.DataFrame(
        {
            "detected_object_id": [0, 1, 2],
            "guide_object_id": [0, 0, 2],
            "detected_object_x_mm": [1.0, 2.0, 3.0],
            "detected_object_y_mm": [-1.0, -2.0, -3.0],
            "guide_object_x_mm": [1.1, 2.1, 3.1],
            "guide_object_y_mm": [-1.1, -2.1, -3.1],
            "detected_object_x_pix": [10.0, 20.0, 30.0],
            "detected_object_y_pix": [11.0, 21.0, 31.0],
            "guide_object_x_pix": [10.5, 20.5, 30.5],
            "guide_object_y_pix": [11.5, 21.5, 31.5],
            "agc_camera_id": [0, 0, 3],
            "matched": [1, 0, 0],
            "camera_enabled": [True, True, True],
            "match_flags": flags,
        }
    )
    return GuideOffsets(
        ra=0.0,
        dec=0.0,
        inst_pa=0.0,
        ra_offset=0.0,
        dec_offset=0.0,
        inr_offset=0.0,
        scale_offset=0.0,
        dalt=None,
        daz=None,
        guide_objects=pd.DataFrame(),
        detected_objects=pd.DataFrame(),
        identified_objects=identified,
        dx=0.0,
        dy=0.0,
        size=0.0,
        peak=0.0,
        flux=0.0,
    )


def test_identified_objects_saves_match_flags_in_place_of_matched(offsets, tmp_path):
    files = offsets.save_numpy_files(base_dir=str(tmp_path))
    (ident_file,) = [f for f in files if f.endswith("identified_objects.npy")]
    ident = np.load(ident_file)

    # Still 12 columns: vgwActor writes the FITS extension with a fixed list of 12 units.
    assert ident.dtype.names == IDENT_COLUMNS
    assert ident["match_flags"].tolist() == offsets.identified_objects["match_flags"].tolist()
    assert ident["match_flags"].dtype.kind == "i"

    # A reader that used to test `matched == 1` gets the same rows from `match_flags == 1`.
    good = ident["match_flags"] == int(F.GOOD_MATCH)
    assert good.tolist() == (offsets.identified_objects["matched"] == 1).tolist()
    # Negative control: the bitmask carries more than matched did, so the columns differ.
    assert ident["match_flags"].tolist() != offsets.identified_objects["matched"].tolist()
    assert "matched" not in ident.dtype.names
