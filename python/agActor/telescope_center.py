import numpy

from pfs_design import pfsDesign as pfs_design


class telCenter:

    def __init__(
        self,
        *,
        actor=None,
        logger=None,
        center=None,
        design=None,
        tel_dither=None,
        tel_guide=None,
        tel_status=None,
    ):

        self._actor = actor
        self._logger = logger if logger is not None else actor.logger if actor is not None else None
        self._center = (
            center
            if center is not None
            else (
                pfs_design(design_id=design[0], design_path=design[1], logger=actor.logger).center
                if design is not None
                else None
            )
        )
        self._tel_dither = tel_dither
        self._tel_guide = tel_guide
        self._tel_status = tel_status

    @property
    def center(self):

        if self._center is not None:
            center = self._center
        else:
            tel_status = self._tel_status if self._tel_status is not None else self._actor.gen2.tel_status
            self._logger and self._logger.info("tel_status={}".format(tel_status))
            tel_guide = self._tel_guide if self._tel_guide is not None else self._actor.gen2.tel_guide
            self._logger and self._logger.info("tel_guide={}".format(tel_guide))
            tel_dither = self._tel_dither if self._tel_dither is not None else self._actor.gen2.tel_dither
            self._logger and self._logger.info("tel_dither={}".format(tel_dither))
            ra, dec, inst_pa = tel_status[
                5:8
            ]  # command ra, dec, and inst_pa (including dither offsets and guide offsets)
            ra -= (
                tel_guide["ra"]
                + tel_dither["ra"] / float(numpy.cos(numpy.deg2rad(dec - tel_guide["dec"] / 3600)))
            ) / 3600
            dec -= (tel_guide["dec"] + tel_dither["dec"]) / 3600
            inst_pa -= (tel_guide["insrot"] + tel_dither["pa"]) / 3600
            center = ra, dec, inst_pa
            self._center = center
        self._logger and self._logger.info("center={}".format(center))
        return center

    @property
    def dither(self):

        tel_guide = self._tel_guide if self._tel_guide is not None else self._actor.gen2.tel_guide
        self._logger and self._logger.info("tel_guide={}".format(tel_guide))
        if self._center is not None:
            tel_dither = self._tel_dither if self._tel_dither is not None else self._actor.gen2.tel_dither
            self._logger and self._logger.info("tel_dither={}".format(tel_dither))
            dec = self._center[1] + tel_dither["dec"] / 3600
            dither = (
                self._center[0] + tel_dither["ra"] / float(numpy.cos(numpy.deg2rad(dec))) / 3600,
                dec,
                self._center[2] + tel_dither["pa"] / 3600,
            )
        else:
            tel_status = self._tel_status if self._tel_status is not None else self._actor.gen2.tel_status
            self._logger and self._logger.info("tel_status={}".format(tel_status))
            dither = (
                tel_status[5] - tel_guide["ra"] / 3600,
                tel_status[6] - tel_guide["dec"] / 3600,
                tel_status[7] - tel_guide["insrot"] / 3600,
            )
        offset = 0, 0, 0, -tel_guide["insrot"]
        self._logger and self._logger.info("dither={},offset={}".format(dither, offset))
        return dither, offset

    @property
    def offset(self):

        tel_guide = self._tel_guide if self._tel_guide is not None else self._actor.gen2.tel_guide
        self._logger and self._logger.info("tel_guide={}".format(tel_guide))
        if self._center is not None:
            tel_dither = self._tel_dither if self._tel_dither is not None else self._actor.gen2.tel_dither
            self._logger and self._logger.info("tel_dither={}".format(tel_dither))
            dec = self._center[1] + tel_dither["dec"] / 3600
            offset = (
                tel_dither["ra"] / float(numpy.cos(numpy.deg2rad(dec))),
                tel_dither["dec"],
                tel_dither["pa"],
                -tel_guide["insrot"],
            )
        else:
            offset = -tel_guide["ra"], -tel_guide["dec"], -tel_guide["insrot"], -tel_guide["insrot"]
        self._logger and self._logger.info("offset={}".format(offset))
        return offset
