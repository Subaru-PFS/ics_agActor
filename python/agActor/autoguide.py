from agActor.catalog import gen2_gaia as gaia, astrometry
from agActor import field_acquisition
from agActor.utils.opdb import opDB as opdb
from agActor.catalog.pfs_design import pfsDesign as pfs_design


class Field:

    design = None
    center = None
    guide_objects = None


def set_design(*, logger=None, **kwargs):

    field_acquisition.parse_kwargs(kwargs)
    design_id = kwargs.get("design_id")
    design_path = kwargs.get("design_path")
    logger and logger.info(f"{design_id=},{design_path=}")
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")
    if any(x is None for x in (ra, dec, inst_pa)):
        if any(x is not None for x in (design_id, design_path)):
            if design_path is not None:
                logger and logger.info(f"Setting psf_design via {design_path=}")
                _ra, _dec, _inst_pa = pfs_design(design_id, design_path, logger=logger).center
                logger and logger.info(f"ra={_ra},dec={_dec},inst_pa={_inst_pa}")
            else:
                logger and logger.info(f"Setting psf_design via {design_id=}")
                _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
                logger and logger.info(f"ra={_ra},dec={_dec},inst_pa={_inst_pa}")
            if ra is None:
                ra = _ra
            if dec is None:
                dec = _dec
            if inst_pa is None:
                inst_pa = _inst_pa
    logger and logger.info(f"ra={ra},dec={dec},inst_pa={inst_pa}")
    logger and logger.info(f"Setting Field.design to {design_id=},{design_path=}")
    Field.design = design_id, design_path
    logger and logger.info(f"Setting Field.center to {ra=},{dec=},{inst_pa=}")
    Field.center = ra, dec, inst_pa
    logger and logger.info(f"Setting Field.guide_objects to []")
    Field.guide_objects = []  # delay loading of guide objects


def set_design_agc(*, frame_id=None, obswl=0.62, logger=None, **kwargs):

    logger and logger.info(f"frame_id={frame_id}")
    field_acquisition.parse_kwargs(kwargs)
    if frame_id is not None:
        logger and logger.info(f"Setting pfs_design_agc via frame_id={frame_id}")
        # generate guide objects from frame
        ra, dec, inst_pa = Field.center
        logger and logger.info(f"ra={ra},dec={dec},inst_pa={inst_pa}")
        taken_at, inr, adc, m2_pos3 = field_acquisition.get_tel_status(
            frame_id=frame_id, logger=logger, **kwargs
        )
        logger and logger.info(f"taken_at={taken_at},inr={inr},adc={adc},m2_pos3={m2_pos3}")
        logger and logger.info("Getting agc_data from oped for frame_id={}".format(frame_id))
        detected_objects = opdb.query_agc_data(frame_id)
        logger and logger.info(f"Got {len(detected_objects)=} detected objects)")

        if "dra" in kwargs:
            ra += kwargs.get("dra") / 3600
        if "ddec" in kwargs:
            dec += kwargs.get("ddec") / 3600
        if "dinr" in kwargs:
            inr += kwargs.get("dinr") / 3600
        logger and logger.info("ra={},dec={},inr={}".format(ra, dec, inr))
        logger and logger.info("Getting guide objects from astrometry")
        guide_objects = astrometry.measure(
            detected_objects=detected_objects,
            ra=ra,
            dec=dec,
            obstime=taken_at,
            inst_pa=inst_pa,
            adc=adc,
            m2_pos3=m2_pos3,
            obswl=obswl,
            logger=logger,
        )
        logger and logger.info(f"Got {len(guide_objects)=} guide objects)")
    else:
        # use guide objects from pfs design file or operational database, or generate on-the-fly
        design_id, design_path = Field.design
        logger and logger.info("design_id={},design_path={}".format(design_id, design_path))
        if design_path is not None:
            logger and logger.info("Getting guide_objects via {}".format(design_path))
            taken_at = kwargs.get("taken_at")

            logger and logger.info("taken_at={}".format(taken_at))
            guide_objects, *_ = pfs_design(design_id, design_path, logger=logger).guide_objects(
                obstime=taken_at
            )
        elif design_id is not None:
            logger and logger.info("Getting guide_objects from opdb via {}".format(design_id))
            guide_objects = opdb.query_pfs_design_agc(design_id)
        else:
            ra, dec, inst_pa = Field.center
            logger and logger.info("ra={},dec={},inst_pa={}".format(ra, dec, inst_pa))
            taken_at = kwargs.get("taken_at")
            inr = kwargs.get("inr")
            adc = kwargs.get("adc")
            m2_pos3 = kwargs.get("m2_pos3", 6.0)
            logger and logger.info("taken_at={},inr={},adc={},m2_pos3={}".format(taken_at, inr, adc, m2_pos3))
            if "dra" in kwargs:
                ra += kwargs.get("dra") / 3600
            if "ddec" in kwargs:
                dec += kwargs.get("ddec") / 3600
            if "dinr" in kwargs:
                inr += kwargs.get("dinr") / 3600
            logger and logger.info("ra={},dec={},inr={}".format(ra, dec, inr))
            logger and logger.info("Getting guide objects from gaia database")
            guide_objects, *_ = gaia.get_objects(
                ra=ra, dec=dec, obstime=taken_at, inst_pa=inst_pa, adc=adc, m2pos3=m2_pos3, obswl=obswl
            )

    logger and logger.info(f"Got {len(guide_objects)} guide objects before filtering.")
    guide_objects = field_acquisition.filter_guide_objects(guide_objects, logger)

    logger and logger.info("Setting Field.guide_objects to")
    Field.guide_objects = guide_objects


def autoguide(*, frame_id, obswl=0.62, logger=None, **kwargs):

    logger and logger.info("Calling autoguide.autoguide with frame_id={}".format(frame_id))
    field_acquisition.parse_kwargs(kwargs)
    guide_objects = Field.guide_objects

    ra, dec, inst_pa = Field.center
    logger and logger.info("ra={},dec={}".format(ra, dec))
    logger and logger.info("Getting telescope status")
    taken_at, inr, adc, m2_pos3 = field_acquisition.get_tel_status(
        frame_id=frame_id, logger=logger, **kwargs
    )
    logger and logger.info("Getting agc_data for frame_id={}".format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id)
    logger and logger.info(f"Got {len(detected_objects)=} detected objects)")
    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
    if "dinr" in kwargs:
        inr += kwargs.get("dinr") / 3600
    logger and logger.info("ra={},dec={},inst_pa={},inr={}".format(ra, dec, inst_pa, inr))
    _kwargs = field_acquisition.filter_kwargs(kwargs)
    logger and logger.info("_kwargs={}".format(_kwargs))
    logger and logger.info("Calling field_acquisition._acquire_field from autoguide.autoguide")
    return (
        ra,
        dec,
        inst_pa,
        *field_acquisition.calculate_guide_offsets(
            guide_objects=guide_objects,
            detected_objects=detected_objects,
            ra=ra,
            dec=dec,
            taken_at=taken_at,
            adc=adc,
            inst_pa=inst_pa,
            m2_pos3=m2_pos3,
            obswl=obswl,
            altazimuth=True,
            logger=logger,
            **_kwargs,
        ),
    )  # (ra, dec, inst_pa, dra, ddec, dinr, dscale, dalt, daz, *values)
