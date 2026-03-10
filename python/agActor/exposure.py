"""Shared AG exposure pipeline.

Encapsulates the common logic for taking an AG camera exposure, gathering
telescope state, computing guide offsets, computing focus, reporting results,
and writing to opDB.

Used by both field acquisition (one-shot) and autoguiding (continuous loop).
See the "Exposure Pipeline" section in README.md for full documentation of the
two operational modes and the pipeline steps.
"""

import logging
import time

import numpy as np

from agActor import autoguide, field_acquisition
from agActor.catalog import pfs_design
from agActor.config import AgConfig
from agActor.utils import actorCalls, data as data_utils
from agActor.utils.actorCalls import sendAlert, send_guide_offsets
from agActor.utils.data import GuideOffsetFlag
from agActor.utils.focus import focus
from agActor.utils.telescope_center import telCenter as tel_center

logger = logging.getLogger(__name__)


def _build_agcc_command(
    exposure_time: int,
    visit_id: int | None,
    exposure_delay: int,
    tec_off: bool,
) -> str:
    """Build the AGCC exposure command string."""
    cmd_str = f"expose object exptime={exposure_time / 1000} centroid=1"
    if visit_id is not None:
        cmd_str += f" visit={visit_id}"
    if exposure_delay > 0:
        cmd_str += f" threadDelay={exposure_delay}"
    if tec_off:
        cmd_str += " tecOFF"
    return cmd_str


def _gather_telescope_state(
    actor,
    cfg: AgConfig,
    visit_id: int | None,
    center,
    design,
    offset,
    kwargs: dict,
):
    """Gather telescope state from mlp1/gen2/opdb during mid-exposure.

    Mutates *kwargs* in-place with telescope state information and returns
    the (possibly updated) *center*, *offset*, and *telescope_state*.
    """
    telescope_state = None

    if cfg.with_mlp1_status:
        telescope_state = actor.mlp1.telescopeState
        logger.info(f"telescopeState={telescope_state}")
        kwargs["inr"] = telescope_state["rotator_real_angle"]

    if cfg.with_gen2_status or cfg.with_opdb_tel_status:
        if cfg.with_gen2_status:
            try:
                tel_status = actorCalls.updateTelStatus(
                    actor, actor.logger, visit_id
                )
            except Exception as e:
                raise RuntimeError(f"updateTelStatus error: {e}")

            logger.info(f"{tel_status=}")
            kwargs["tel_status"] = tel_status
            _tel_center = tel_center(
                actor=actor,
                center=center,
                design=design,
                tel_status=tel_status,
            )
            if all(x is None for x in (center, design)):
                center, _offset = _tel_center.dither
                logger.info(f"{center=}")
            else:
                _offset = _tel_center.offset
            if offset is None:
                offset = _offset
                logger.info(f"{offset=}")

        if cfg.with_opdb_tel_status:
            status_update = actor.gen2.statusUpdate
            status_id = (status_update["visit"], status_update["sequenceNum"])
            logger.info(f"status_id={status_id}")
            kwargs["status_id"] = status_id

    return center, offset, telescope_state


def _compute_taken_at(
    actor,
    cfg: AgConfig,
    exposure_time: int,
    exposure_delay: int,
    telescope_state,
    kwargs: dict,
) -> float:
    """Compute the ``taken_at`` timestamp after exposure completes.

    Mutates *kwargs* with the ``taken_at`` key when appropriate and returns
    the final ``taken_at`` value.
    """
    data_time = actor.agcc.dataTime
    logger.info(f"dataTime={data_time}")
    taken_at = data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2
    logger.info(f"{taken_at=}")

    if cfg.with_agcc_timestamp:
        kwargs["taken_at"] = taken_at  # unix timestamp, not timezone-aware datetime
    if cfg.with_mlp1_status:
        taken_at = actor.mlp1.setUnixDay(
            telescope_state["az_el_detect_time"], taken_at
        )
        kwargs["taken_at"] = taken_at

    return taken_at


def _report_guide_offsets(cmd, guide_offsets):
    """Emit the standard cmd.inform messages for guide offsets."""
    ra = guide_offsets.ra
    dec = guide_offsets.dec
    inst_pa = guide_offsets.inst_pa
    dra = guide_offsets.ra_offset
    ddec = guide_offsets.dec_offset
    dinr = guide_offsets.inr_offset
    dscale = guide_offsets.scale_offset
    dalt = guide_offsets.dalt
    daz = guide_offsets.daz

    cmd.inform(
        f'text="{ra=},{dec=},{inst_pa=},{dra=},{ddec=},{dinr=},{dscale=},{dalt=},{daz=}"'
    )

    filenames = guide_offsets.save_numpy_files()
    cmd.inform(
        'data={},{},{},"{}","{}","{}"'.format(ra, dec, inst_pa, *filenames)
    )


def _compute_focus(guide_offsets, max_ellipticity, max_size, min_size):
    """Compute focus offset and tilt from matched detected objects.

    Uses only matched detected objects (those with ``matched == 1``).
    """
    matched_ids = (
        guide_offsets.identified_objects
        .query("matched == 1")
        .detected_object_id
        .values
    )
    # Use only matched detected objects for focus computation.
    detected = guide_offsets.detected_objects
    if len(matched_ids) > 0:
        detected = detected.loc[matched_ids]

    return focus(
        detected_objects=detected,
        max_ellipticity=max_ellipticity,
        max_size=max_size,
        min_size=min_size,
    )


def run_exposure_pipeline(
    *,
    actor,
    cmd,
    cfg: AgConfig,
    design_id: int | None,
    design_path: str | None,
    design,
    visit_id: int | None,
    visit0: int | None,
    exposure_time: int,
    exposure_delay: int,
    tec_off: bool,
    center=None,
    offset=None,
    dinr=None,
    guide_catalog=None,
    send_offsets: bool = True,
    dry_run: bool = False,
    max_correction: float | None = None,
    max_ellipticity: float = 2.0,
    max_size: float = 1.0e12,
    min_size: float = -1.0,
    **kwargs,
):
    """Run the full AG exposure-to-correction pipeline.

    This is the single entry point for the shared logic between
    ``AgCmd.acquire_field`` (one-shot field acquisition) and ``AgThread.run``
    (continuous autoguiding loop).

    The two modes are distinguished primarily by the ``guide_catalog`` parameter:

    * **Field acquisition** (``guide_catalog=None``): The pipeline calls
      ``field_acquisition.acquire_field`` which fetches its own guide catalog
      (with ``is_guide=False``) and detected objects for this single frame.
      Typically called with ``max_correction=None`` (no range checking) since
      the initial acquisition offset can be large.

    * **Autoguiding** (``guide_catalog=<GuideCatalog>``): The pipeline calls
      ``autoguide.get_exposure_offsets`` using a pre-loaded guide catalog
      (fetched once with ``is_guide=True`` before the guide loop begins).
      Only telescope status and detected objects are re-fetched per frame.
      Called with a ``max_correction`` value to reject unreasonably large
      corrections.

    Parameters
    ----------
    actor : AgActor
        The actor instance for queuing commands and accessing models.
    cmd
        The command object for sending ``cmd.inform`` messages.
    cfg : AgConfig
        Shared AG configuration flags.
    design_id : int or None
        Design ID for guide star lookup.
    design_path : str or None
        Path to the pfsDesign file.
    design : tuple or None
        ``(design_id, design_path)`` tuple, or None.
    visit_id : int or None
        Visit ID for the exposure.
    visit0 : int or None
        The visit0 for guide star lookup from pfs_config_agc.
    exposure_time : int
        Exposure time in milliseconds.
    exposure_delay : int
        Exposure delay in milliseconds.
    tec_off : bool
        Whether to turn off TEC.
    center : tuple or None
        Field center as (ra, dec[, pa]).
    offset : tuple or None
        Field offset as (dra, ddec[, dpa[, dinr]]).
    dinr : float or None
        Instrument rotator offset.
    guide_catalog : GuideCatalog or None
        Pre-loaded guide catalog for autoguiding (fetched with ``is_guide=True``).
        If provided, uses ``autoguide.get_exposure_offsets`` which reuses this
        catalog across frames.  If None, uses ``field_acquisition.acquire_field``
        which fetches its own catalog (with ``is_guide=False``) for one-shot
        field acquisition.
    send_offsets : bool
        Whether to send guide offsets to the telescope.  Defaults to True.
    dry_run : bool
        If True, don't actually send corrections.
    max_correction : float or None
        Maximum allowed correction in arcsec.  If None, no range checking.
    max_ellipticity : float
        Maximum ellipticity for source filtering.
    max_size : float
        Maximum size for source filtering.
    min_size : float
        Minimum size for source filtering.
    **kwargs
        Additional keyword arguments passed through to the offset computation
        (e.g. ``magnitude``, ``fit_dinr``, ``fit_dscale``, ``max_residual``,
        ``filter_bad_shape``).

    Returns
    -------
    GuideOffsets
        The computed guide offsets.
    """
    # --- 1. Take the AGCC exposure ---
    cmd.inform(f"exposureTime={exposure_time}")
    cmd_str = _build_agcc_command(exposure_time, visit_id, exposure_delay, tec_off)
    logger.info(f"Sending agcc {cmd_str=}")

    agcc_exposure_result = actor.queueCommand(
        actor="agcc",
        cmdStr=cmd_str,
        timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 15),
    )

    # Sleep until roughly mid-exposure to gather telescope state.
    time.sleep((exposure_time + 7 * exposure_delay) / 1000 / 2)

    # --- 2. Gather telescope state mid-exposure ---
    center, offset, telescope_state = _gather_telescope_state(
        actor, cfg, visit_id, center, design, offset, kwargs
    )

    # --- 3. Wait for exposure to complete ---
    agcc_exposure_result.get()
    frame_id = actor.agcc.frameId
    logger.info(f"{frame_id=}")

    # --- 4. Compute taken_at ---
    taken_at = _compute_taken_at(
        actor, cfg, exposure_time, exposure_delay, telescope_state, kwargs
    )

    # Pack remaining positional params into kwargs for offset computation.
    if center is not None:
        kwargs["center"] = center
    if offset is not None:
        kwargs["offset"] = offset
    if dinr is not None:
        kwargs["dinr"] = dinr

    # --- 5. Compute guide offsets ---
    cmd.inform("detectionState=1")

    if guide_catalog is not None:
        # Autoguide path: use the pre-loaded guide catalog (is_guide=True).
        # Only telescope status and detected objects are re-fetched per frame.
        max_residual = kwargs.pop("max_residual", 0.5)
        logger.info(f"Computing offsets via autoguide for {frame_id=}")
        guide_offsets = autoguide.get_exposure_offsets(
            frame_id=frame_id,
            guide_catalog=guide_catalog,
            max_ellipticity=max_ellipticity,
            max_size=max_size,
            min_size=min_size,
            max_residual=max_residual,
            **kwargs,
        )
    else:
        # Acquire-field path: fetch guide catalog internally (is_guide=False)
        # for this single frame.  The initial offset can be large.
        logger.info(f"Computing offsets via field_acquisition for {frame_id=}")
        guide_offsets = field_acquisition.acquire_field(
            design_id=design_id,
            visit0=visit0,
            frame_id=frame_id,
            **kwargs,
        )

    # --- 6. Report results ---
    _report_guide_offsets(cmd, guide_offsets)
    cmd.inform("detectionState=0")

    ra = guide_offsets.ra
    dec = guide_offsets.dec
    inst_pa = guide_offsets.inst_pa
    dra = guide_offsets.ra_offset
    ddec = guide_offsets.dec_offset
    dinr_offset = guide_offsets.inr_offset
    dscale = guide_offsets.scale_offset
    dalt = guide_offsets.dalt
    daz = guide_offsets.daz

    # --- 7. Range checking and sending guide offsets ---
    offset_flags = GuideOffsetFlag.OK
    guide_status = "OK"

    if max_correction is not None:
        offset_in_range = abs(dra) < max_correction and abs(ddec) < max_correction
        if not offset_in_range:
            offset_flags = GuideOffsetFlag.INVALID_OFFSET
            guide_status = "INVALID_OFFSET"

    if send_offsets and offset_flags == GuideOffsetFlag.OK:
        send_guide_offsets(
            actor=actor,
            taken_at=taken_at,
            daz=daz,
            dalt=dalt,
            dx=guide_offsets.dx,
            dy=guide_offsets.dy,
            size=guide_offsets.size,
            peak=guide_offsets.peak,
            flux=guide_offsets.flux,
            dry_run=dry_run,
            logger=actor.logger,
        )
    elif max_correction is not None and offset_flags != GuideOffsetFlag.OK:
        cmd.inform(
            f'text="Calculated offset not in allowed range, skipping: {dra=} {ddec=} {max_correction=}"'
        )
        sendAlert(
            actor=actor,
            alert_id="AG.OFFSET_OUT_OF_RANGE",
            alert_name="Autoguide Offset Out of Range",
            alert_description="The calculated autoguide offset is out of the allowed range, no corrections have been sent to the telescope.",
            alert_detail=f"Calculated offsets: {frame_id=} {visit_id=} {dra=}, {ddec=}, {max_correction=}",
            alert_severity="warning",
            logger=actor.logger,
        )

    # --- 8. Focus ---
    logger.info(f"Computing focus for {frame_id=}")
    dz, dzs = _compute_focus(guide_offsets, max_ellipticity, max_size, min_size)

    if dalt is None:
        dalt = np.nan
    if daz is None:
        daz = np.nan

    cmd.inform(
        "guideErrors={},{},{},{},{},{},{},{},{}".format(
            frame_id, dra, ddec, dinr_offset, daz, dalt, dz, dscale, guide_status
        )
    )
    cmd.inform("focusErrors={},{},{},{},{},{},{}".format(frame_id, *dzs))

    # --- 9. Write to opDB ---
    if cfg.with_opdb_agc_guide_offset:
        logger.info(f"Writing agc_guide_offset for {frame_id=}")
        data_utils.write_agc_guide_offset(
            frame_id=frame_id,
            ra=ra,
            dec=dec,
            pa=inst_pa,
            delta_ra=dra,
            delta_dec=ddec,
            delta_insrot=dinr_offset,
            delta_scale=dscale,
            delta_az=daz,
            delta_el=dalt,
            delta_z=dz,
            delta_zs=dzs,
            offset_flags=offset_flags,
        )
    if cfg.with_opdb_agc_match:
        logger.info(f"Writing agc_match for {frame_id=}")
        _design_id = design_id
        if _design_id is None and design_path is not None:
            _design_id = pfs_design.pfsDesign.to_design_id(design_path)
        if _design_id is None:
            _design_id = 0
        data_utils.write_agc_match(
            design_id=_design_id,
            frame_id=frame_id,
            guide_objects=guide_offsets.guide_objects,
            detected_objects=guide_offsets.detected_objects,
            identified_objects=guide_offsets.identified_objects,
        )

    return guide_offsets
