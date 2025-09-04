from datetime import datetime, timezone
from numbers import Number

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Angle, Distance, EarthLocation, SkyCoord
from astropy.time import Time
from pfs.utils.coordinates import Subaru_POPT2_PFS, coordinates

_popt2 = Subaru_POPT2_PFS.POPT2()
_pfs = Subaru_POPT2_PFS.PFS()


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
    icam = np.mod(np.rint(np.arctan2(y_dp, x_dp) * 3 / np.pi), 6).astype(np.intc)
    x_det, y_det = coordinates.dp2det(icam, x_dp, y_dp)
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
    tanz = np.tan(np.deg2rad(z))
    y_adc = e0 + (e1 + (e2 + (e3 + e4 * tanz) * tanz) * tanz) * tanz
    return np.clip(y_adc, 0, 22)


def process_search_results(
    objects, obstime, altaz_c, frame_tc, adc, inr, m2pos3=6.0, obswl=0.62
):
    """
    Process search results to apply space motion and convert coordinates.

    Parameters
    ----------
    objects : astropy.table.Table
        The table of Gaia DR3 sources from search
    obstime : astropy.time.Time
        The time of the observation
    altaz_c : astropy.coordinates.SkyCoord
        The field center in AltAz frame
    frame_tc : astropy.coordinates.AltAz
        The AltAz frame for the observation
    adc : scalar
        The position of the atmospheric dispersion compensator (mm)
    inr : scalar
        The instrument rotator angle, east of north (deg)
    m2pos3 : scalar
        The z position of the hexapod (mm)
    obswl : scalar
        The wavelength of the observation (um)

    Returns
    -------
    numpy.ndarray
        A structured array containing guide objects with fields:
        - source_id: Gaia source identifier
        - ra, dec: coordinates in degrees
        - mag: magnitude
        - camera_id: camera identifier
        - x, y: detector coordinates
        - x_dp, y_dp: detector plane coordinates
        - x_fp, y_fp: focal plane coordinates
    """
    objects["parallax"][np.where(objects["parallax"] < 1e-6)] = 1e-6
    _icrs = SkyCoord(
        ra=objects["ra"],
        dec=objects["dec"],
        frame="icrs",
        distance=Distance(parallax=Angle(objects["parallax"])),
        pm_ra_cosdec=objects["pmra"],
        pm_dec=objects["pmdec"],
        obstime=Time(objects["ref_epoch"], format="jyear", scale="tcb"),
    )
    _icrs_d = _icrs.apply_space_motion(new_obstime=obstime)  # of date
    _altaz = _icrs_d.transform_to(frame_tc)
    separation = altaz_c.separation(_altaz).to(u.deg).value
    position_angle = altaz_c.position_angle(_altaz).to(u.deg).value
    flags = 0  # Generic search, so no flags.
    x_fp, y_fp = _popt2.celestial2focalplane(
        separation,
        -position_angle,
        adc,
        inr,
        altaz_c.alt.to(u.deg).value,
        m2pos3,
        obswl,
        flags,
    )
    x_dp, y_dp = _pfs.fp2dp(x_fp, y_fp, inr)
    icam, x_det, y_det = dp2idet(x_dp, y_dp)

    # Create structured array directly from the data
    n_objects = len(objects)

    rows = list()
    for i in range(n_objects):
        if x_det[i] < 0 or y_det[i] < 0:
            continue

        rows.append(
            [
                objects[i]["source_id"],
                _icrs_d.ra[i].to(u.deg).value,
                _icrs_d.dec[i].to(u.deg).value,
                objects[i]["phot_g_mean_mag"],
                icam[i],
                x_det[i],
                y_det[i],
                0,  # Fake flag
            ]
        )

    return rows


def setup_search_coordinates(
    ra,
    dec,
    obstime=None,
    cameras=None,
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
    Set up proper search coordinates for guide stellar objects.

    Parameters
    ----------
    ra : astropy.coordinates.Angle or float
        The right ascension (ICRS) of the field center (deg)
    dec : astropy.coordinates.Angle or float
        The declination (ICRS) of the field center (deg)
    obstime : astropy.time.Time, datetime, or float
        The time of the observation
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
    tuple
        A tuple containing:
        - icrs: SkyCoord object with search coordinates
        - altaz_c: SkyCoord object with field center in AltAz frame
        - frame_tc: AltAz frame for the observation
        - inr: instrument rotator angle
        - adc: atmospheric dispersion compensator position
    """
    if cameras is None:
        cameras = [0, 1, 2, 3, 4, 5]

    ra = Angle(ra, unit=u.deg)
    dec = Angle(dec, unit=u.deg)
    obstime = (
        Time(obstime.astimezone(tz=timezone.utc))
        if isinstance(obstime, datetime)
        else (
            Time(obstime, format="unix")
            if isinstance(obstime, Number)
            else Time(obstime)
            if obstime is not None
            else Time.now()
        )
    )

    # Get Subaru location
    subaru_location = EarthLocation.from_geodetic(
        lon=-155.476111 * u.deg, lat=19.825556 * u.deg, height=4139 * u.m
    )

    frame_tc = AltAz(
        obstime=obstime,
        location=subaru_location,
        temperature=temperature * u.deg_C,
        relative_humidity=relative_humidity / 100,
        pressure=pressure * u.hPa,
        obswl=obswl * u.micron,
    )

    # field center
    icrs_c = SkyCoord(ra=ra, dec=dec, frame="icrs")
    altaz_c = icrs_c.transform_to(frame_tc)

    if inr is None:
        # celestial north pole
        icrs_p = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs")
        altaz_p = icrs_p.transform_to(frame_tc)
        parallactic_angle = altaz_c.position_angle(altaz_p).to(u.deg).value
        inr = (parallactic_angle + inst_pa + 180) % 360 - 180

    # centers of the detectors in the detector plane coordinates
    x_dp, y_dp = coordinates.det2dp(np.asarray(cameras), (511.5 + 24), (511.5 + 9))

    # centers of the detectors in the focal plane coordinates
    x_fp, y_fp = _pfs.dp2fp(x_dp, y_dp, inr)

    if adc is None:
        if filter_id is None:
            filter_id = 107  # wideband, uniform weighting
        adc = z2adc(altaz_c.zen.to(u.deg).value, filter_id=filter_id)  # mm

    flags = 0  # Generic search, so no flags.
    separation, position_angle = _popt2.focalplane2celestial(
        x_fp, y_fp, adc, inr, altaz_c.alt.to(u.deg).value, m2pos3, obswl, flags
    )
    altaz = altaz_c.directional_offset_by(-position_angle * u.deg, separation * u.deg)
    icrs = altaz.transform_to("icrs")

    return icrs, altaz_c, frame_tc, inr, adc
