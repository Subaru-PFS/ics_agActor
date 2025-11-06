from agActor.main import AgActor


def send_guide_offsets(
    *, actor, taken_at, daz, dalt, dx, dy, size, peak, flux, dry_run, logger
):
    """Send guider offsets to mlp1.

    Parameters
    ----------
    actor : `AgActor`
       where to send commands and fetch results from.
    taken_at : float
        Timestamp when the image was taken (MJD).
    daz : float
        Azimuth offset in arcseconds (positive feedback).
    dalt : float
        Altitude offset in arcseconds (positive feedback).
    dx : float
        X offset in arcseconds (HSC -> PFS).
    dy : float
        Y offset in arcseconds (HSC -> PFS).
    size : float
        Representative spot size in mas.
    peak : float
        Representative peak intensity in adu.
    flux : float
        Representative flux in adu.
    dry_run : bool
        If True, do not apply the corrections.
    logger : `logging.Logger`
        Logger to use.

    Returns
    -------
    mlp1_result : result of the mlp1 command
    """
    logger.info(
        f"Calling mlp1.guide with caller={actor.name}, {taken_at=}, {daz=}, {dalt=}, {dx=}, {dy=}, {size=}, {peak=}, {flux=}, {dry_run=}"
    )

    # send corrections to mlp1 and gen2 (or iic)
    mlp1_result = actor.queueCommand(
        actor="mlp1",
        # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
        cmdStr="guide azel={},{} ready={} time={} delay=0 xy={},{} size={} intensity={} flux={}".format(
            -daz,
            -dalt,
            int(not dry_run),
            taken_at,
            dx * 1e3,
            -dy * 1e3,
            size,
            peak,
            flux,
        ),
        timeLim=5,
    ).get()

    return mlp1_result


def updateTelStatus(actor, logger, visit_id=None):
    """Fetch the current tel_status.

    Parameters
    ----------
    actor : `AgActor`
       where to send commands and fetch results from.
    logger : `logging.Logger`
        Logger to use.
    visit_id : int, optional
       if we know it, the visit to associate status with

    Returns
    -------
    tel_status : the gen2.tel_status keyword value
    """

    # Note that visit_id can be 0 when the ag is testing or doing OTF
    if visit_id:
        visitStr = f"visit={visit_id}"
    else:
        visitStr = ""

    logger.info(
        f"Calling gen2.updateTelStatus with caller={actor.name} and {visitStr=}"
    )

    actor.queueCommand(
        actor="gen2",
        cmdStr=f"updateTelStatus caller={actor.name} {visitStr}",
        timeLim=10,
    ).get()
    tel_status = actor.gen2.tel_status

    return tel_status
