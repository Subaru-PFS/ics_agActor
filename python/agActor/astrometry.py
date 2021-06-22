import itertools
import numpy
from astropy import units
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import coordinates
from kawanomoto import Subaru_POPT2_PFS


iers.conf.auto_download = True
solar_system_ephemeris.set('de430')

popt2 = Subaru_POPT2_PFS.POPT2()
pfs = Subaru_POPT2_PFS.PFS()


def measure(
        detected_objects,
        ra,
        dec,
        obstime,
        inr,
        adc,
        temperature=0,
        relative_humidity=0,
        pressure=620,
        m2_pos3=0.55,
        obswl=0.62,
        logger=None
):

    logger and logger.info('ra={},dec={},obstime={},inr={},adc={},m2_pos3={},obswl={}'.format(ra, dec, obstime, inr, adc, m2_pos3, obswl))

    ra = Angle(ra, unit=units.deg)
    dec = Angle(dec, unit=units.deg)
    obstime = Time(obstime)

    # subaru coordinates (NAD83 ~ WGS 1984 at 0.1" level, height of elevation axis)
    location = EarthLocation(lat=Angle((19, 49, 31.8), unit=units.deg), lon=Angle((-155, 28, 33.7), unit=units.deg), height=4163)
    frame_tc = AltAz(obstime=obstime, location=location, temperature=temperature * units.deg_C, relative_humidity=relative_humidity / 100, pressure=pressure * units.hPa, obswl=obswl * units.micron)

    # field center in the horizontal coordinates
    icrs_c = SkyCoord(ra=ra, dec=dec, frame='icrs')
    altaz_c = icrs_c.transform_to(frame_tc)

    # detected stellar objects in the equatorial coordinates
    icam, x_det, y_det, flags = numpy.array(detected_objects)[:, (0, 3, 4, -1)].T
    x_dp, y_dp = coordinates.det2dp(numpy.rint(icam - 1), x_det, y_det)
    x_fp, y_fp = pfs.dp2fp(x_dp, y_dp, inr)
    separation, position_angle = popt2.focalplane2celestial(x_fp, y_fp, adc, m2_pos3, obswl, flags)
    altaz = altaz_c.directional_offset_by(- position_angle * units.deg, separation * units.deg)
    icrs = altaz.transform_to('icrs')

    # source_id, ra, dec, mag
    counter = itertools.count(1)
    mag = 0
    objects = [(next(counter), x.ra.to(units.deg).value, x.dec.to(units.deg).value, mag) for x in icrs]

    return objects


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--tile-id', type=int, required=True, help='tile identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    args, _ = parser.parse_known_args()

    from opdb import opDB as opdb

    ra, dec, _ = opdb.query_tile(args.tile_id)
    _, _, taken_at, _, _, inr, adc, temperature, relative_humidity, pressure, m2_pos3 = opdb.query_agc_exposure(args.frame_id)
    detected_objects = opdb.query_agc_data(args.frame_id)

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='astrometry')

    objects = measure(
        detected_objects=detected_objects,
        ra=ra,
        dec=dec,
        obstime=taken_at,
        inr=inr,
        adc=adc,
        temperature=temperature,
        relative_humidity=relative_humidity,
        pressure=pressure,
        m2_pos3=m2_pos3,
        obswl=args.obswl,
        logger=logger
    )
    print(objects)
