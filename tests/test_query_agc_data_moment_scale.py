"""query_agc_data returns second moments on one scale, the spots' variance.

ics_agccActor stored half the variance while it wrote the literal "git" as its version; those
exposures are doubled as they are read. The query is not run here: the tests check that it
asks for the scaling, and with which values.
"""

import re

from agActor.utils import data
from pfs.utils.datamodel.ag import SourceDetectionFlag

MOMENTS = ("central_image_moment_11_pix", "central_image_moment_20_pix", "central_image_moment_02_pix")


def _captured_query(monkeypatch) -> tuple[str, dict]:
    captured = {}

    def fake_query_db(sql, params, **kwargs):
        captured.update(sql=sql, params=params)

    monkeypatch.setattr(data, "query_db", fake_query_db)
    data.query_agc_data(1234)
    return captured["sql"], captured["params"]


def test_moments_are_scaled_by_the_exposures_agcc_version(monkeypatch):
    sql, params = _captured_query(monkeypatch)

    assert params["agc_exposure_id"] == 1234
    # Exposures marked "git" hold half the variance; BAD_SHAPE spots hold unweighted moments.
    assert params["halved_scale"] == 2.0
    assert params["bad_shape"] == int(SourceDetectionFlag.BAD_SHAPE) == 8
    assert re.search(r"version_actor = 'git' AND COALESCE\(agc_data\.flags, 0\) & :bad_shape = 0\s+THEN :halved_scale ELSE 1", sql)
    for moment in MOMENTS:
        assert f"agc_data.{moment} * scale.variance_per_moment AS {moment}" in sql
        # Negative control: no moment is selected as stored.
        assert not re.search(rf"agc_data\.{moment},", sql)
    # Negative control: nothing else is scaled.
    assert sql.count("scale.variance_per_moment") == len(MOMENTS)
