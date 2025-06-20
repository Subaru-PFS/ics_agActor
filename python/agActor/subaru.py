from astropy import units
from astropy.coordinates import Angle, EarthLocation

# location = EarthLocation.of_site('Subaru Telescope')

# subaru coordinates (NAD83 ~ WGS 1984 at 0.1" level, height of elevation axis)
height = elevation = 4163 * units.m
latitude = Angle("+19:49:31.8", unit=units.deg)
longitude = Angle("-155:28:33.7", unit=units.deg)
location = EarthLocation(lat=latitude, lon=longitude, height=height)
