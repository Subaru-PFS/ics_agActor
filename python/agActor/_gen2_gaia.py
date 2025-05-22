from datetime import datetime, timezone
from numbers import Number
import numpy
from astropy import units
from astropy.coordinates import AltAz, Angle, Distance, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import coordinates

from pfs.utils.coordinates import Subaru_POPT2_PFS


iers.conf.auto_download = True
solar_system_ephemeris.set('de440')

_popt2 = Subaru_POPT2_PFS.POPT2()
_pfs = Subaru_POPT2_PFS.PFS()


def sky2fp(separation, position_angle, adc, inr, alt, m2pos3=6.0, obswl=0.62, flag=0):
    """
    Convert angular offsets to focal plane coordinates.

    Convert angular separations and position angles of points from the field
    center on the celestial sphere to Cartesian coordinates on the focal plane
    in the focal plane coordinate system (which is affixed to the telescope).

    Parameters
    ----------
    separation : array_like
        The angular separations of points from the field center (deg)
    position_angle : array_like
        The position angles, west of north, of points relative to the field
        center (deg)
    adc : scalar
        The position of the atmospheric dispersion compensator (mm)
    m2pos3 : scalar
        The z position of the hexapod (mm)
    obswl : scalar
        The wavelength of the observation (um)
    flag : array_like
        The flags indicating whether points are on the far focus side (false)
        or on the near focus side (true) of the detector

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane
        in the focal plane coordinate system (mm)
    """

    return _popt2.celestial2focalplane(separation, position_angle, adc, inr, alt, m2pos3, obswl, flag)


def fp2sky(x_fp, y_fp, adc, inr, alt, m2pos3=6.0, obswl=0.62, flag=0):
    """
    Convert focal plane coordinates to angular offsets.

    Convert Cartesian coordinates of points on the focal plane in the focal
    plane coordinate system (which is affixed to the telescope) to angular
    separations and position angles from the field center on the celestial
    sphere.

    Parameters
    ----------
    x_fp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        focal plane coordinate system (mm)
    y_fp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        focal plane coordinate system (mm)
    adc : scalar
        The position of the atmospheric dispersion compensator (mm)
    m2pos3 : scalar
        The z position of the hexapod (mm)
    obswl : scalar
        The wavelength of the observation (um)
    flag : array_like
        The flags indicating whether points are on the far focus side (false)
        or on the near focus side (true) of the detector

    Returns
    -------
    2-tuple of array_likes
        The angular separations and position angles, west of north, of points
        from the field center on the celestial sphere (deg)
    """

    return _popt2.focalplane2celestial(x_fp, y_fp, adc, inr, alt, m2pos3, obswl, flag)


def fp2dp(x_fp, y_fp, inr):
    """
    Convert focal plane coordinates to detector plane coordinates.

    Convert Cartesian coordinates of points on the focal plane in the focal
    plane coordinate system (which is affixed to the telescope) to those in the
    detector plane coordinate system (which corotates with the instrument
    rotator).

    Parameters
    ----------
    x_fp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        focal plane coordinate system (mm)
    y_fp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        focal plane coordinate system (mm)
    inr : scalar
        The instrument rotator angle, east of north (deg)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane in
        the detector plane coordinate system (mm)
    """

    return _pfs.fp2dp(x_fp, y_fp, inr)


def dp2fp(x_dp, y_dp, inr):
    """
    Convert detector plane coordinates to focal plane coordinates.

    Convert Cartesian coordinates of points on the focal plane in the detector
    plane coordinate system (which corotates with the instrument rotator) to
    those in the focal plane coordinate system (which is affixed to the
    telescope).

    Parameters
    ----------
    x_dp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        detector plane coordinate system (mm)
    y_dp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        detector plane coordinate system (mm)
    inr : scalar
        The instrument rotator angle, east of north (deg)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane in
        the focal plane coordinate system (mm)
    """

    return _pfs.dp2fp(x_dp, y_dp, inr)


def dp2det(icam, x_dp, y_dp):
    """
    Convert detector plane coordinates to detector coordinates.

    Convert Cartesian coordinates of points on the focal plane in the detector
    plane coordinate system to those on one of the detectors in the detector
    coordinate system.

    Parameters
    ----------
    icam : array_like
        The detector identifiers ([0, 5])
    x_dp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        detector plane coordinate system (mm)
    y_dp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        detector plane coordinate system (mm)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the specified
        detector in the detector coordinate system (pix)
    """

    return coordinates.dp2det(icam, x_dp, y_dp)


def det2dp(icam, x_det, y_det):
    """
    Convert detector coordinates to detector plane coordinates.

    Convert Cartesian coordinates of points on one of the detectors in the
    detector coordinate system to those on the focal plane in the detector
    plane coordinate system.

    Parameters
    ----------
    icam : array_like
        The detector identifiers ([0, 5])
    x_det : array_like
        The Cartesian coordinates x's of points on the specified detector in
        the detector coordinate system (pix)
    y_det : array_like
        The Cartesian coordinates y's of points on the specified detector in
        the detector coordinate system (pix)

    Returns
    -------
    2-tuple of array_likes
        The Cartesian coordinates x's and y's of points on the focal plane in
        the detector plane coordinate system (mm)
    """

    return coordinates.det2dp(icam, x_det, y_det)


def dp2idet(x_dp, y_dp):
    """
    Convert detector plane coordinates to detector coordinates.

    Convert Cartesian coordinates of points on the focal plane in the detector
    plane coordinate system to those on the detectors in the detector
    coordinate system. The detector for each point is determined by its
    position angle.

    Parameters
    ----------
    x_dp : array_like
        The Cartesian coordinates x's of points on the focal plane in the
        detector plane coordinate system (mm)
    y_dp : array_like
        The Cartesian coordinates y's of points on the focal plane in the
        detector plane coordinate system (mm)

    Returns
    -------
    3-tuple of array_likes
        The detector identifiers and Cartesian coordinates x's and y's of
        points on the detectors in the detector coordinate system (pix)
    """

    # determine the detector id from the position angle of the detector plane coordinates
    icam = numpy.mod(numpy.rint(numpy.arctan2(y_dp, x_dp) * 3 / numpy.pi), 6).astype(numpy.intc)
    x_det, y_det = dp2det(icam, x_dp, y_dp)
    return icam, x_det, y_det


def z2adc(z, filter_id):
    """
    Compute ADC positions from telescope zenith distances.

    Compute positions of the atmospheric dispersion compensator from zenith
    distances of the telescope.

    Parameters
    ----------
    z : array_like
        The zenith distances of the telescope (deg)
    filter_id : scalar
        The filter identifier (101-108)

    Returns
    -------
    array_like
        The positions of the atmospheric dispersion compensator (mm)
    """

    # from Table 6.4.2-2 of S1503702001C
    _ADC_PARAMS = {
        # key: filter identifier
        # values: wavelength, a, b, c, d, e0, e1, e2, e3, e4
        101: (260, 1, 0, 0, 0, -0.0524, 1.63248e1, -1.5774e1, 1.44935e1, -4.2412),
        102: (315, 1, 0, 0, 0, 0.1446, 3.5804, 1.78643e1, -8.7045, 8.03e-1),
        103: (305, 1, 0, 0, 0, 0.1963, -2.4117, 2.57883e1, -2.36341e1, 6.3793),
        104: (575, 1, 0, 0, 0, -0.0081, 1.25224e1, -7.141e-1, 6.026e-1, -1.71e-1),
        105: (620, 1, 0, 0, 0, -0.3981, 1.29418e1, 7.131e-1, -1.3105, 3.957e-1),
        106: (986, 1, 0, 0, 0, 0.03, 1.20713e1, 1.1829, -9.201e-1, 2.37e-1),
        107: (986, 1, 0, 0, 0, -0.0897, 1.39256e1, -4.5109, 3.6753, -9.312e-1),
        108: (986, 1, 0, 0, 0, -0.0792, 1.25646e1, 4.19e-2, -3.699e-1, -2.111e-1),
    }

    _, _, _, _, _, e0, e1, e2, e3, e4 = _ADC_PARAMS[filter_id]
    tanz = numpy.tan(numpy.deg2rad(z))
    y_adc = e0 + (e1 + (e2 + (e3 + e4 * tanz) * tanz) * tanz) * tanz
    return numpy.clip(y_adc, 0, 22)


def search(ra, dec, radius=0.027 + 0.003):
    """
    Search guide stellar objects from Gaia DR3 sources.

    Parameters
    ----------
    ra : array_like
        The right ascensions (ICRS) of the search centers (deg)
    dec : array_like
        The declinations (ICRS) of the search centers (deg)
    radius : scalar
        The radius of the cones (deg)

    Returns
    -------
    astropy.table.Table
        The table of the Gaia DR3 sources inside the search areas
    """

    def _search(ra, dec, radius):
        """Perform search of Gaia DR3."""

        if numpy.isscalar(ra):
            ra = (ra,)
        if numpy.isscalar(dec):
            dec = (dec,)

        import psycopg2
        from astropy.table import Table

        columns = ('source_id', 'ref_epoch', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'phot_g_mean_mag')
        _units = (units.dimensionless_unscaled, units.yr, units.deg, units.mas, units.deg, units.mas, units.mas, units.mas, units.mas / units.yr, units.mas / units.yr, units.mas / units.yr, units.mas / units.yr, units.mag)

        host = '133.40.167.46'  # 'g2db' for production use
        port = 5438
        user = 'gen2'  # 'obsuser' for production use

        dsn = 'host={} port={} user={} dbname=star_catalog'.format(host, port, user)
        with psycopg2.connect(dsn) as connection:
            with connection.cursor() as cursor:
                query = 'SELECT {} FROM gaia3 WHERE ('.format(','.join(columns)) \
                    + ' OR '.join(['q3c_radial_query(ra,dec,{},{},{})'.format(_ra, _dec, radius) for _ra, _dec in zip(ra, dec)]) \
                    + ') AND pmra IS NOT NULL AND pmdec IS NOT NULL AND parallax IS NOT NULL ORDER BY phot_g_mean_mag'
                cursor.execute(query)
                objects = cursor.fetchall()
                return Table(rows=objects, names=columns, units=_units)

    return _search(ra, dec, radius)


def get_objects(
        ra,
        dec,
        obstime=None,
        cameras=[0, 1, 2, 3, 4, 5],
        inst_pa=0,
        inr=None,
        adc=None,
        filter_id=None,
        temperature=0,
        relative_humidity=0,
        pressure=620,
        obswl=0.62,
        m2pos3=6.0,
):
    """
    Get list of guide stellar objects.

    Parameters
    ----------
    ra : astropy.coordinates.Angle
        The right ascension (ICRS) of the field center (deg)
    dec : astropy.coordinates.Angle
        The declination (ICRS) of the field center (deg)
    obstime : astropy.time.Time
        The time of the observation (datetime)
    cameras : sequence
        The sequence of the camera identifiers (0-5)
    inst_pa : scalar
        The position angle of the instrument, east of north (deg)
    inr : scalar
        The instrument rotator angle, east of north (deg)
    adc : scalar, None
        The position of the atmospheric dispersion compensator (mm)
    filter_id : scalar
        The filter identifier for ADC (101-108)
    temperature : scalar
        The air temperature (deg C)
    relative_humidity : scalar
        The relative humidity (%)
    pressure : scalar
        The atmospheric pressure (hPa)
    obswl : scalar
        The wavelength of the observation (um)
    m2pos3 : scalar
        The z position of the hexapod (mm)

    Returns
    -------
    list of 5-tuples
        The list of the tuples of camera identifier, object identifier,
        detector coordinate x, detector coordinate y, and mean magnitude of the
        guide stellar objects
    """

    ra = Angle(ra, unit=units.deg)
    dec = Angle(dec, unit=units.deg)
    obstime = Time(obstime.astimezone(tz=timezone.utc)) if isinstance(obstime, datetime) else Time(obstime, format='unix') if isinstance(obstime, Number) else Time(obstime) if obstime is not None else Time.now()

    import subaru
    frame_tc = AltAz(obstime=obstime, location=subaru.location, temperature=temperature * units.deg_C, relative_humidity=relative_humidity / 100, pressure=pressure * units.hPa, obswl=obswl * units.micron)

    # field center
    icrs_c = SkyCoord(ra=ra, dec=dec, frame='icrs')
    altaz_c = icrs_c.transform_to(frame_tc)

    if inr is None:
        # celestial north pole
        icrs_p = SkyCoord(ra=0 * units.deg, dec=90 * units.deg, frame='icrs')
        altaz_p = icrs_p.transform_to(frame_tc)
        parallactic_angle = altaz_c.position_angle(altaz_p).to(units.deg).value
        inr = (parallactic_angle + inst_pa + 180) % 360 - 180

    # centers of the detectors in the detector plane coordinates
    x_dp, y_dp = det2dp(numpy.asarray(cameras), (511.5 + 24), (511.5 + 9))

    # centers of the detectors in the focal plane coordinates
    x_fp, y_fp = dp2fp(x_dp, y_dp, inr)

    if adc is None:
        if filter_id is None:
            filter_id = 107  # wideband, uniform weighting
        adc = z2adc(altaz_c.zen.to(units.deg).value, filter_id=filter_id)  # mm

    separation, position_angle = fp2sky(x_fp, y_fp, adc, inr, altaz_c.alt.to(units.deg).value, m2pos3, obswl)
    altaz = altaz_c.directional_offset_by(- position_angle * units.deg, separation * units.deg)
    icrs = altaz.transform_to('icrs')

    _objects = search(icrs.ra.deg, icrs.dec.deg)
    _objects['parallax'][numpy.where(_objects['parallax'] < 1e-6)] = 1e-6
    _icrs = SkyCoord(
        ra=_objects['ra'], dec=_objects['dec'], frame='icrs',
        distance=Distance(parallax=Angle(_objects['parallax'])),
        pm_ra_cosdec=_objects['pmra'], pm_dec=_objects['pmdec'],
        obstime=Time(_objects['ref_epoch'], format='jyear', scale='tcb')
    )
    _icrs_d = _icrs.apply_space_motion(new_obstime=obstime)  # of date
    _altaz = _icrs_d.transform_to(frame_tc)
    separation = altaz_c.separation(_altaz).to(units.deg).value
    position_angle = altaz_c.position_angle(_altaz).to(units.deg).value
    x_fp, y_fp = sky2fp(separation, - position_angle, adc, inr, altaz_c.alt.to(units.deg).value, m2pos3, obswl)
    x_dp, y_dp = fp2dp(x_fp, y_fp, inr)
    icam, x_det, y_det = dp2idet(x_dp, y_dp)

    objects = numpy.array(
        [
            (
                _source_id,
                _skycoord.ra.to(units.deg).value,
                _skycoord.dec.to(units.deg).value,
                _mag,
                _camera_id,
                _x_det,
                _y_det,
                _x_dp,
                _y_dp,
                _x_fp,
                _y_fp
            )
            for _source_id, _skycoord, _mag, _camera_id, _x_det, _y_det, _x_dp, _y_dp, _x_fp, _y_fp
            in zip(
                _objects['source_id'],
                _icrs_d,
                _objects['phot_g_mean_mag'],
                icam,
                x_det,
                y_det,
                x_dp,
                y_dp,
                x_fp,
                y_fp
            )
        ],
        dtype=[
            ('source_id', numpy.int64),  # u8 (80) not supported by FITSIO
            ('ra', numpy.float64),
            ('dec', numpy.float64),
            ('mag', numpy.float32),
            ('camera_id', numpy.int16),
            ('x', numpy.float32),
            ('y', numpy.float32),
            ('x_dp', numpy.float32),
            ('y_dp', numpy.float32),
            ('x_fp', numpy.float32),
            ('y_fp', numpy.float32)
        ]
    )

    return objects, altaz_c.az.to(units.deg).value, altaz_c.alt.to(units.deg).value, inr, adc


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('ra', help='right ascension (ICRS) of the field center (hr)')
    parser.add_argument('dec', help='declination (ICRS) of the field center (deg)')
    parser.add_argument('obstime', nargs='?', default=None, help='time of observation (datetime)')
    parser.add_argument('--cameras', default='1,2,3,4,5,6', help='comma-separated camera identifiers (1-6)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--inst-pa', type=float, default=0, help='position angle of the instrument, east of north (deg)')
    group.add_argument('--inr', type=float, default=None, help='instrument rotator angle, east of north (deg)')
    parser.add_argument('--adc', type=float, default=None, help='position of the atmospheric dispersion compensator (mm)')
    parser.add_argument('--filter-id', type=int, default=107, help='filter identifier for ADC (101-108)')
    parser.add_argument('--temperature', type=float, default=0, help='air temperature (deg C)')
    parser.add_argument('--relative-humidity', type=float, default=0, help='relative humidity (%%)')
    parser.add_argument('--pressure', type=float, default=620, help='atmospheric pressure (hPa)')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--m2-pos3', type=float, default=6.0, help='z position of the hexapod (mm)')

    args, _ = parser.parse_known_args()

    objects, az, alt, inr, adc = get_objects(
        ra=15 * Angle(args.ra, unit=units.hourangle).value,
        dec=args.dec,
        obstime=args.obstime,
        cameras=[int(x) - 1 for x in args.cameras.split(',')],
        inst_pa=args.inst_pa,
        inr=args.inr,
        adc=args.adc,
        filter_id=args.filter_id,
        temperature=args.temperature,
        relative_humidity=args.relative_humidity,
        pressure=args.pressure,
        obswl=args.obswl,
        m2pos3=args.m2_pos3,
    )
    print(objects)
    print(az, alt, inr, adc)
