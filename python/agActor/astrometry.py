import itertools
from datetime import datetime, timezone
from numbers import Number

import numpy
from astropy import units
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time

from agActor import coordinates
from agActor import subaru
from agActor.kawanomoto import Subaru_POPT2_PFS

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
    logger=None
):

    logger and logger.info(
        f'{ra=},{dec=},{obstime=},{inst_pa=},{inr=},{adc=},{m2_pos3=},{temperature=},{relative_humidity=},'
        f'{pressure=},{obswl=}'
    )

    ra = Angle(ra, unit=units.deg)
    dec = Angle(dec, unit=units.deg)

    if isinstance(obstime, datetime):
        obstime = Time(obstime.astimezone(tz=timezone.utc))
    elif isinstance(obstime, Number):
        obstime = Time(obstime, format='unix')
    elif obstime is not None:
        obstime = Time(obstime)
    else:
        obstime = Time.now()

    frame_tc = AltAz(
        obstime=obstime,
        location=subaru.location,
        temperature=temperature * units.deg_C,
        relative_humidity=relative_humidity / 100,
        pressure=pressure * units.hPa,
        obswl=obswl * units.micron
    )
    # field center in the horizontal coordinates
    icrs_c = SkyCoord(ra=ra, dec=dec, frame='icrs')
    altaz_c = icrs_c.transform_to(frame_tc)

    if inr is None:
        # celestial north pole
        icrs_p = SkyCoord(ra=0 * units.deg, dec=90 * units.deg, frame='icrs')
        altaz_p = icrs_p.transform_to(frame_tc)
        parallactic_angle = altaz_c.position_angle(altaz_p).to(units.deg).value
        inr = (parallactic_angle + inst_pa + 180) % 360 - 180
        logger and logger.info(f'{parallactic_angle=},{inst_pa=},{inr=}')

    # detected stellar objects in the equatorial coordinates
    icam, x_detected, y_detected, flags = numpy.array(detected_objects)[:, (0, 3, 4, -1)].T
    x_dp, y_dp = coordinates.det2dp(numpy.rint(icam), x_detected, y_detected)
    x_fp, y_fp = pfs.dp2fp(x_dp, y_dp, inr)
    _, alt = _subaru.radec2azel(ra, dec, obswl, obstime)
    separation, position_angle = popt2.focalplane2celestial(x_fp, y_fp, adc, inr, alt, m2_pos3, obswl, flags)
    altaz = altaz_c.directional_offset_by(-position_angle * units.deg, separation * units.deg)
    icrs = altaz.transform_to('icrs')

    # source_id, ra, dec, mag
    counter = itertools.count()
    mag = 0
    objects = numpy.array(
        [
            (
                next(counter),
                coord.ra.to(units.deg).value,
                coord.dec.to(units.deg).value,
                mag,
                cam_id
            )
            for coord, cam_id
            in zip(icrs, icam)
        ],
        dtype=[
            ('source_id', numpy.int64),  # u8 (80) not supported by FITSIO
            ('ra', numpy.float64),
            ('dec', numpy.float64),
            ('mag', numpy.float32),
            ('camera_id', numpy.int16),
            # ('guide_object_xdet', numpy.float32),
            # ('guide_object_ydet', numpy.float32)
        ]
    )

    return objects


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), required=True, help='design identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    args, _ = parser.parse_known_args()

    from opdb import opDB as opdb

    _, ra, dec, inst_pa, *_ = opdb.query_pfs_design(args.design_id)
    _, _, taken_at, _, _, _, adc, temperature, relative_humidity, pressure, m2_pos3 = opdb.query_agc_exposure(
        args.frame_id
    )
    detected_objects = opdb.query_agc_data(args.frame_id)

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='astrometry')
    objects = measure(
        detected_objects=detected_objects,
        ra=ra,
        dec=dec,
        obstime=taken_at,
        inst_pa=inst_pa,
        adc=adc,
        m2_pos3=m2_pos3,
        temperature=temperature,
        relative_humidity=relative_humidity,
        pressure=pressure,
        obswl=args.obswl,
        logger=logger
    )
    print(objects)
