"""AgCmd builds from an actor config that leaves the optional entries out."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from agActor.Commands import AgCmd as agCmdModule


def _ag_cmd(actor_config: dict) -> agCmdModule.AgCmd:
    actor = MagicMock()
    actor.actorConfig = actor_config
    actor.logger = logging.getLogger("test_ag")
    with (
        patch.object(agCmdModule.OpDB, "set_default_connection"),
        patch.object(agCmdModule.GaiaDB, "set_default_connection"),
    ):
        return agCmdModule.AgCmd(actor)


def test_tel_status_defaults_to_the_agc_exposure_row():
    cmd_set = _ag_cmd({})
    assert (cmd_set.with_gen2_status, cmd_set.with_mlp1_status, cmd_set.with_opdb_tel_status) == (False, False, False)


@pytest.mark.parametrize(
    ("tel_status", "expected"),
    [("gen2,mlp1", (True, True, False)), ("gen2, tel_status", (True, False, True))],
)
def test_tel_status_entry_selects_the_sources(tel_status, expected):
    # Negative control for the default above: a configured entry does switch the sources on.
    cmd_set = _ag_cmd({"tel_status": tel_status})
    assert (cmd_set.with_gen2_status, cmd_set.with_mlp1_status, cmd_set.with_opdb_tel_status) == expected
