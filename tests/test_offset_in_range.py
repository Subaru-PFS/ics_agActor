"""max_correction applies to the offset on the sky, not to the raw RA coordinate offset (INSTRM-2931)."""

import numpy as np
import pytest
from agActor.autoguide import offset_in_range

DEC = 66.0  # the field of 2026-05-19; cos = 0.407
COS = np.cos(np.deg2rad(DEC))


def test_ra_limit_is_on_the_sky():
    # 13.6 arcsec of RA coordinate at Dec 66 is 5.5 arcsec on the sky: within a 10 arcsec limit.
    assert offset_in_range(13.6, 0.1, DEC, 10.0)
    # Negative control: compared raw, the same offset is rejected.
    assert not (abs(13.6) < 10.0 and abs(0.1) < 10.0)
    # And 30 arcsec of RA coordinate (12 on the sky) is rejected either way.
    assert not offset_in_range(30.0, 0.1, DEC, 10.0)


def test_dec_has_no_cos_factor():
    assert not offset_in_range(0.1, 10.5, DEC, 10.0)
    assert offset_in_range(0.1, 9.5, DEC, 10.0)


@pytest.mark.parametrize("dec", [-30.0, 0.0, 45.0, 66.0, 80.0])
def test_limit_is_the_same_sky_distance_at_every_declination(dec):
    cos = np.cos(np.deg2rad(dec))
    assert offset_in_range(9.9 / cos, 0.0, dec, 10.0)
    assert not offset_in_range(10.1 / cos, 0.0, dec, 10.0)
    # Negative control: the raw comparison accepts that offset only at the equator.
    assert (abs(9.9 / cos) < 10.0) == (dec == 0.0)


def test_sign_does_not_matter():
    assert offset_in_range(-13.6, -0.1, DEC, 10.0)
    assert not offset_in_range(-30.0, 0.0, DEC, 10.0)
