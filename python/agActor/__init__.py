from astropy.coordinates import solar_system_ephemeris
from astropy.utils import iers

iers.conf.auto_download = True
solar_system_ephemeris.set('de440')
