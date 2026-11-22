"""The seeing log line: the image size corrected to zenith with the instrument PSF removed."""
import math

import pandas as pd
import pytest
from pfs.utils import agMetrics
from pfs.utils.agMetrics import AgImageSize

from agActor.utils.data import GuideOffsets


def _offsets(image_size):
    """A GuideOffsets carrying only the image size; the rest is irrelevant to describe_seeing."""
    empty = pd.DataFrame()
    return GuideOffsets(
        ra=0.0, dec=0.0, inst_pa=0.0, ra_offset=0.0, dec_offset=0.0, inr_offset=0.0,
        scale_offset=0.0, dalt=None, daz=None, guide_objects=empty, detected_objects=empty,
        identified_objects=empty, dx=0.0, dy=0.0,
        size=(image_size.fwhm if image_size else 0.0), peak=0.0, flux=0.0, image_size=image_size,
    )


def test_describe_seeing_corrects_to_zenith_and_removes_the_psf():
    """At zenith the airmass is 1, so the seeing is the image size with the PSF out in quadrature."""
    offsets = _offsets(AgImageSize(fwhm=0.9, fwhmLeft=0.9, fwhmRight=0.9, nLeft=5, nRight=5))
    line = offsets.describe_seeing(1234, 90.0)
    assert "frame_id=1234 seeing=" in line and "arcsec (zenith)" in line
    seeing = float(line.split("seeing=")[1].split()[0])
    assert seeing == pytest.approx(math.sqrt(0.9**2 - agMetrics.AG_SEEING_PSF_FWHM**2), abs=5e-4)
    # Negative control: the calibrated seeing is below the raw image size (the PSF is removed)...
    assert seeing < 0.9
    # ...and away from zenith the airmass correction lowers it further.
    lower = float(offsets.describe_seeing(1234, 40.0).split("seeing=")[1].split()[0])
    assert lower < seeing


def test_describe_seeing_not_measured_cases():
    """No image size, no altitude, or a size at or below the PSF reports not measured."""
    assert "not measured" in _offsets(None).describe_seeing(1, 90.0)
    psf = agMetrics.AG_SEEING_PSF_FWHM
    at_psf = AgImageSize(fwhm=psf, fwhmLeft=psf, fwhmRight=psf, nLeft=1, nRight=1)
    assert "not measured" in _offsets(at_psf).describe_seeing(1, None)
    assert "at or below the instrument PSF" in _offsets(at_psf).describe_seeing(1, 90.0)
