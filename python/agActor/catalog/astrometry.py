import itertools
from datetime import datetime, timezone
from numbers import Number

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Angle, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
from pfs.utils.coordinates import Subaru_POPT2_PFS, coordinates

from agActor.utils.logging import log_message

iers.conf.auto_download = True
solar_system_ephemeris.set("de440")

_subaru = Subaru_POPT2_PFS.Subaru()
popt2 = Subaru_POPT2_PFS.POPT2()
pfs = Subaru_POPT2_PFS.PFS()


def measure(
    detected_objects,
    ra,
    dec,
    obstime=None,
    inst_pa=0,
    inr=None,
    adc=0,
    m2_pos3=6.0,
    temperature=0,
    relative_humidity=0,
    pressure=620,
    obswl=0.62,
    logger=None,
):

    log_message(
        logger,
        f"{ra=},{dec=},{obstime=},{inst_pa=},{inr=},{adc=},"
        f"{m2_pos3=},{temperature=},{relative_humidity=},{pressure=},{obswl=}",
    )

    ra = Angle(ra, unit=u.deg)
    dec = Angle(dec, unit=u.deg)
    obstime = (
        Time(obstime.astimezone(tz=timezone.utc))
        if isinstance(obstime, datetime)
        else (
            Time(obstime, format="unix")
            if isinstance(obstime, Number)
            else Time(obstime) if obstime is not None else Time.now()
        )
    )

    from agActor.utils import subaru

    frame_tc = AltAz(
        obstime=obstime,
        location=subaru.location,
        temperature=temperature * u.deg_C,
        relative_humidity=relative_humidity / 100,
        pressure=pressure * u.hPa,
        obswl=obswl * u.micron,
    )
    # field center in the horizontal coordinates
    icrs_c = SkyCoord(ra=ra, dec=dec, frame="icrs")
    altaz_c = icrs_c.transform_to(frame_tc)

    if inr is None:
        # celestial north pole
        icrs_p = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs")
        altaz_p = icrs_p.transform_to(frame_tc)
        parallactic_angle = altaz_c.position_angle(altaz_p).to(u.deg).value
        inr = (parallactic_angle + inst_pa + 180) % 360 - 180
        log_message(logger, f"parallactic_angle={parallactic_angle},inst_pa={inst_pa},inr={inr}")

    # detected stellar objects in the equatorial coordinates
    icam, x_det, y_det, flags = np.array(detected_objects)[:, (0, 3, 4, -1)].T
    x_dp, y_dp = coordinates.det2dp(np.rint(icam), x_det, y_det)
    x_fp, y_fp = pfs.dp2fp(x_dp, y_dp, inr)
    _, alt = _subaru.radec2azel(ra, dec, obswl, obstime)
    separation, position_angle = popt2.focalplane2celestial(x_fp, y_fp, adc, inr, alt, m2_pos3, obswl, flags)
    altaz = altaz_c.directional_offset_by(-position_angle * u.deg, separation * u.deg)
    icrs = altaz.transform_to("icrs")

    # source_id, ra, dec, mag
    counter = itertools.count()
    mag = 0
    objects = np.array(
        [(next(counter), x.ra.to(u.deg).value, x.dec.to(u.deg).value, mag) for x in icrs],
        dtype=[
            ("source_id", np.int64),  # u8 (80) not supported by FITSIO
            ("ra", np.float64),
            ("dec", np.float64),
            ("mag", np.float32),
        ],
    )

    return objects
