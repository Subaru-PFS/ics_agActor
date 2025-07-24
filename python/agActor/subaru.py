from astropy import units as u
from astropy.coordinates import Angle, EarthLocation

# subaru coordinates (NAD83 ~ WGS 1984 at 0.1" level, height of elevation axis)
height = elevation = 4163 * u.m
latitude = Angle("+19:49:31.8", unit=u.deg)
longitude = Angle("-155:28:33.7", unit=u.deg)
location = EarthLocation(lat=latitude, lon=longitude, height=height)
