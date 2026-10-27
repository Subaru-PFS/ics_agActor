"""The detection flags that keep an AG spot out of acquisition and guiding."""

from agActor.utils.data import BAD_DETECTION_FLAGS
from pfs.utils.datamodel.ag import SourceDetectionFlags as F


def test_bad_detection_flags():
    for flag in (F.SATURATED, F.EDGE, F.BAD_ELLIP):
        assert BAD_DETECTION_FLAGS & flag, flag
    # Negative control: RIGHT is the glass-free half of the detector, not a defect,
    # and FLAT_TOP is retired (INSTRM-2586), so neither may drop a spot.
    assert BAD_DETECTION_FLAGS & F.RIGHT == 0
    assert BAD_DETECTION_FLAGS & F.FLAT_TOP == 0
