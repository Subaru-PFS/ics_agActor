"""The command send_guide_offsets builds for mlp1."""

import logging

import pytest
from agActor.utils.actorCalls import send_guide_offsets


class _Queued:
    def get(self):
        return "result"


class _Actor:
    name = "ag"

    def __init__(self):
        self.calls = []

    def queueCommand(self, **kwargs):
        self.calls.append(kwargs)
        return _Queued()


def _keywords(cmd_str):
    return dict(word.split("=") for word in cmd_str.split()[1:])


def test_units_of_the_command_for_mlp1():
    """The xy error goes to mlp1 in mas; the spot size in arcsec, as it becomes the SEEING header card."""
    actor = _Actor()
    result = send_guide_offsets(
        actor=actor, taken_at=1.5, daz=0.2, dalt=-0.1, dx=0.03, dy=0.05, size=0.71, peak=280.0, flux=13400.0,
        dry_run=False, logger=logging.getLogger("test"),
    )  # fmt: skip

    assert result == "result"
    (call,) = actor.calls
    assert call["actor"] == "mlp1"
    keywords = _keywords(call["cmdStr"])
    assert float(keywords["size"]) == pytest.approx(0.71)
    assert [float(v) for v in keywords["xy"].split(",")] == pytest.approx([30.0, -50.0])
    assert [float(v) for v in keywords["azel"].split(",")] == pytest.approx([-0.2, 0.1])  # arcsec
    assert keywords["ready"] == "1"
    # Negative control: the size is not converted to mas as the xy error is.
    assert float(keywords["size"]) != pytest.approx(710.0, rel=0.5)
