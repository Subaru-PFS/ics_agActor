# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
from datetime import datetime, timezone
from numbers import Number

from astropy import units
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time

_KEYMAP = {
    'fit_dinr': ('inrflag', int),
    'fit_dscale': ('scaleflag', int),
    'max_ellipticity': ('maxellip', float),
    'max_size': ('maxsize', float),
    'min_size': ('minsize', float),
    'max_residual': ('maxresid', float)
}


def parse_kwargs(kwargs):

    if (center := kwargs.pop('center', None)) is not None:
        ra, dec, *optional = center
        kwargs.setdefault('ra', ra)
        kwargs.setdefault('dec', dec)
        kwargs.setdefault('inst_pa', optional[0] if len(optional) > 0 else 0)
    if (offset := kwargs.pop('offset', None)) is not None:
        dra, ddec, *optional = offset
        kwargs.setdefault('dra', dra)
        kwargs.setdefault('ddec', ddec)
        kwargs.setdefault('dpa', optional[0] if len(optional) > 0 else kwargs.get('dinr', 0))
        kwargs.setdefault('dinr', optional[1] if len(optional) > 1 else optional[0] if len(optional) > 0 else 0)
    if (design := kwargs.pop('design', None)) is not None:
        design_id, design_path = design
        kwargs.setdefault('design_id', design_id)
        kwargs.setdefault('design_path', design_path)
    if (status_id := kwargs.pop('status_id', None)) is not None:
        visit_id, sequence_id = status_id
        kwargs.setdefault('visit_id', visit_id)
        kwargs.setdefault('sequence_id', sequence_id)
    if (tel_status := kwargs.pop('tel_status', None)) is not None:
        _, _, inr, adc, m2_pos3, _, _, _, taken_at = tel_status
        kwargs.setdefault('taken_at', taken_at)
        kwargs.setdefault('inr', inr)
        kwargs.setdefault('adc', adc)
        kwargs.setdefault('m2_pos3', m2_pos3)


def filter_kwargs(kwargs):

    return {k: v for k, v in kwargs.items() if k in _KEYMAP}


def map_kwargs(kwargs):

    return {_KEYMAP[k][0]: _KEYMAP[k][1](v) for k, v in kwargs.items() if k in _KEYMAP}


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

    ra = Angle(ra, unit=units.deg)
    dec = Angle(dec, unit=units.deg)
    obstime = Time(obstime.astimezone(tz=timezone.utc)) if isinstance(obstime, datetime) else Time(obstime, format='unix') if isinstance(obstime, Number) else Time(obstime) if obstime is not None else Time.now()

    temperature *= units.deg_C
    relative_humidity /= 100
    pressure *= units.hPa
    obswl *= units.micron

    dra *= units.arcsec
    ddec *= units.arcsec

    import subaru
    frame = AltAz(obstime=obstime, location=subaru.location, temperature=temperature, relative_humidity=relative_humidity, pressure=pressure, obswl=obswl)

    icrs = SkyCoord(ra=[ra, ra + dra], dec=[dec, dec + ddec], frame='icrs')
    altaz = icrs.transform_to(frame)

    alt = altaz[0].alt.to(units.deg).value
    az = altaz[0].az.to(units.deg).value

    dalt = (altaz[1].alt - altaz[0].alt).to(units.arcsec).value
    daz = (altaz[1].az - altaz[0].az).to(units.arcsec).value

    return alt, az, dalt, daz
