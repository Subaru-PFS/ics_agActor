from datetime import datetime, timezone
from numbers import Number

from astropy import units as u
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time

from pfs.utils.location import SUBARU


def to_altaz(
    ra, dec, obstime=None, temperature=0, relative_humidity=0, pressure=620, obswl=0.62, dra=0, ddec=0
):

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

    temperature *= u.deg_C
    relative_humidity /= 100
    pressure *= u.hPa
    obswl *= u.micron

    dra *= u.arcsec
    ddec *= u.arcsec

    frame = AltAz(
        obstime=obstime,
        location=SUBARU.location,
        temperature=temperature,
        relative_humidity=relative_humidity,
        pressure=pressure,
        obswl=obswl,
    )

    icrs = SkyCoord(ra=[ra, ra + dra], dec=[dec, dec + ddec], frame="icrs")
    altaz = icrs.transform_to(frame)

    alt = altaz[0].alt.to(u.deg).value
    az = altaz[0].az.to(u.deg).value

    dalt = (altaz[1].alt - altaz[0].alt).to(u.arcsec).value
    daz = (altaz[1].az - altaz[0].az).to(u.arcsec).value

    return alt, az, dalt, daz
