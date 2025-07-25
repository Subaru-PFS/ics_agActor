import os
from datetime import datetime, timezone
from numbers import Number

import fitsio
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Distance, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers

from agActor.utils.logging import log_message

iers.conf.auto_download = True
solar_system_ephemeris.set("de440")


class pfsDesign:

    def __init__(self, design_id=None, design_path=None, logger=None):

        if design_id is None:
            if design_path is None:
                raise TypeError('__init__() missing argument(s): "design_id" and/or "design_path"')
            elif not os.path.isfile(design_path):
                raise ValueError(f'__init__() "design_path" not an existing regular file: "{design_path}"')
        else:
            if design_path is None:
                design_path = "/data/pfsDesign"
            elif not os.path.isdir(design_path):
                raise ValueError(f'__init__() "design_path" not an existing directory: "{design_path}"')
            design_path = self.to_design_path(design_id, design_path)
        self.design_path = design_path
        self.logger = logger

    @property
    def center(self):

        with fitsio.FITS(self.design_path) as fits:
            header = fits[0].read_header()
            ra = header["RA"]
            dec = header["DEC"]
            inst_pa = header["POSANG"]
        return ra, dec, inst_pa

    def guide_objects(self, obstime=None):

        _obstime = (
            Time(obstime.astimezone(tz=timezone.utc))
            if isinstance(obstime, datetime)
            else (
                Time(obstime, format="unix")
                if isinstance(obstime, Number)
                else Time(obstime) if obstime is not None else Time.now()
            )
        )
        log_message(self.logger, f"obstime={obstime},_obstime={_obstime}")
        with fitsio.FITS(self.design_path) as fits:
            header = fits[0].read_header()
            ra = header["RA"]
            dec = header["DEC"]
            inst_pa = header["POSANG"]
            _guide_objects = fits["guidestars"].read()

        _guide_objects["parallax"][np.where(_guide_objects["parallax"] < 1e-6)] = 1e-6
        _icrs = SkyCoord(
            ra=_guide_objects["ra"] * u.deg,
            dec=_guide_objects["dec"] * u.deg,
            frame="icrs",
            distance=Distance(parallax=Angle(_guide_objects["parallax"], unit=u.mas)),
            pm_ra_cosdec=_guide_objects["pmRa"] * u.mas / u.yr,
            pm_dec=_guide_objects["pmDec"] * u.mas / u.yr,
            obstime=Time(_guide_objects["epoch"], format="jyear_str", scale="tcb"),
        )
        _icrs_d = _icrs.apply_space_motion(new_obstime=_obstime)  # of date
        _guide_objects["ra"] = _icrs_d.ra.deg
        _guide_objects["dec"] = _icrs_d.dec.deg

        cols = ["objId", "ra", "dec", "magnitude", "agId", "agX", "agY"]

        # Check if 'flag' column exists and if so, add it to cols.
        if "flag" in _guide_objects.dtype.names:
            cols.append("flag")

        guide_objects = _guide_objects[cols]

        return guide_objects, ra, dec, inst_pa

    @staticmethod
    def to_design_id(design_path):

        filename = os.path.splitext(os.path.basename(design_path))[0]
        return int(filename[10:], 0) if filename.startswith("pfsDesign-") else 0

    @staticmethod
    def to_design_path(design_id, design_path=""):

        return os.path.join(design_path, f"pfsDesign-0x{design_id:016x}.fits")
