"""Shared configuration for the AG actor.

Parsed once from ``actorConfig`` and stored on the actor instance so that
both ``AgCmd`` and ``AgThread`` can access the same values without
duplicating the parsing logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgConfig:
    """Immutable snapshot of AG-relevant actor configuration flags."""

    with_opdb_agc_guide_offset: bool = False
    with_opdb_agc_match: bool = False
    with_agcc_timestamp: bool = False
    with_gen2_status: bool = False
    with_mlp1_status: bool = False
    with_opdb_tel_status: bool = False
    with_design_path: str | None = None

    @classmethod
    def from_actor_config(cls, actor_config: dict) -> "AgConfig":
        """Build an ``AgConfig`` from the raw ``actorConfig`` dict."""
        tel_status_raw = [
            x.strip()
            for x in actor_config.get("tel_status", "agc_exposure").split(",")
        ]
        design_path = actor_config.get("design_path", "").strip() or None

        return cls(
            with_opdb_agc_guide_offset=actor_config.get("agc_guide_offset", False),
            with_opdb_agc_match=actor_config.get("agc_match", False),
            with_agcc_timestamp=actor_config.get("agcc_timestamp", False),
            with_gen2_status="gen2" in tel_status_raw,
            with_mlp1_status="mlp1" in tel_status_raw,
            with_opdb_tel_status="tel_status" in tel_status_raw,
            with_design_path=design_path,
        )
