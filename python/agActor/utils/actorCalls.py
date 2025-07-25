def updateTelStatus(actor, logger, visit_id=None):
    """Fetch the current tel_status.

    Parameters
    ----------
    actor : `actorcore.Actor`
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

    logger.info(f"Calling gen2.updateTelStatus with caller={actor.name} and {visitStr=}")

    actor.queueCommand(
        actor="gen2", cmdStr=f"updateTelStatus caller={actor.name} {visitStr}", timeLim=5
    ).get()
    tel_status = actor.gen2.tel_status

    return tel_status
