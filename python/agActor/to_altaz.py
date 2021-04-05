from astropy import units
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers


iers.conf.auto_download = True
solar_system_ephemeris.set('de430')


def to_altaz(
        ra,
        dec,
        obstime=None,
        temperature=0,
        relative_humidity=0,
        pressure=620,
        obswl=0.62,
        dra=0,
        ddec=0
):

    ra = Angle(ra, unit=units.hourangle)
    dec = Angle(dec, unit=units.deg)
    obstime = Time(obstime) if obstime is not None else Time.now()

    temperature *= units.deg_C
    relative_humidity /= 100
    pressure *= units.hPa
    obswl *= units.micron

    dra *= units.arcsec
    ddec *= units.arcsec

    # subaru coordinates (NAD83 ~ WGS 1984 at 0.1" level, height of elevation axis)
    location = EarthLocation(lat=Angle((19, 49, 31.8), unit=units.deg), lon=Angle((-155, 28, 33.7), unit=units.deg), height=4163)
    frame = AltAz(obstime=obstime, location=location, temperature=temperature, relative_humidity=relative_humidity, pressure=pressure, obswl=obswl)

    icrs = SkyCoord(ra=[ra, ra + dra], dec=[dec, dec + ddec], frame='icrs')
    altaz = icrs.transform_to(frame)

    alt = altaz[0].alt.to(units.deg).value
    az = altaz[0].az.to(units.deg).value

    dalt = (altaz[1].alt - altaz[0].alt).to(units.arcsec).value
    daz = (altaz[1].az - altaz[0].az).to(units.arcsec).value

    return alt, az, dalt, daz


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('ra', help='right ascension (ICRS) of the field center (hr)')
    parser.add_argument('dec', help='declination (ICRS) of the field center (deg)')
    parser.add_argument('obstime', nargs='?', default=None, help='time of observation (datetime)')
    parser.add_argument('--temperature', type=float, default=0, help='air temperature (deg C)')
    parser.add_argument('--relative-humidity', type=float, default=0, help='relative humidity (%%)')
    parser.add_argument('--pressure', type=float, default=620, help='atmospheric pressure (hPa)')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--dra', type=float, default=0, help='(")')
    parser.add_argument('--ddec', type=float, default=0, help='(")')
    args, _ = parser.parse_known_args()

    alt, az, dalt, daz = to_altaz(
        ra=args.ra,
        dec=args.dec,
        obstime=args.obstime,
        temperature=args.temperature,
        relative_humidity=args.relative_humidity,
        pressure=args.pressure,
        obswl=args.obswl,
        dra=args.dra,
        ddec=args.ddec
    )

    print(alt, az, dalt, daz)
