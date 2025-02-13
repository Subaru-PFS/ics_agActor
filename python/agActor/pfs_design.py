import os
from datetime import datetime, timezone
from numbers import Number

import fitsio
from astropy import units
from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.time import Time
from numpy._typing import ArrayLike


class pfsDesign:

    def __init__(self, design_id=None, design_path=None, logger=None):

        if design_id is None:
            if design_path is None:
                raise TypeError('__init__() missing argument(s): "design_id" and/or "design_path"')
            elif not os.path.isfile(design_path):
                raise ValueError('__init__() "design_path" not an existing regular file: "{}"'.format(design_path))
        else:
            if design_path is None:
                design_path = '/data/pfsDesign'
            elif not os.path.isdir(design_path):
                raise ValueError('__init__() "design_path" not an existing directory: "{}"'.format(design_path))
            design_path = self.to_design_path(design_id, design_path)
        self.design_path = design_path
        self.logger = logger

    @property
    def center(self):

        with fitsio.FITS(self.design_path) as fits:
            header = fits[0].read_header()
            ra = header['RA']
            dec = header['DEC']
            inst_pa = header['POSANG']
        return ra, dec, inst_pa

    def get_guide_objects(self, taken_at: datetime | Number | str | None = None) -> tuple[ArrayLike | list, float, float, float]:
        if isinstance(taken_at, datetime):
            _obstime = Time(taken_at.astimezone(tz=timezone.utc))
        elif isinstance(taken_at, Number):
            _obstime = Time(taken_at, format='unix')
        elif taken_at is not None:
            _obstime = Time(taken_at)
        else:
            _obstime = Time.now()
        self.logger and self.logger.info(f"obstime={taken_at},_obstime={_obstime}")

        with fitsio.FITS(self.design_path) as fits:
            header = fits[0].read_header()
            ra = header['RA']
            dec = header['DEC']
            inst_pa = header['POSANG']
            guide_objects = fits['guidestars'].read()

        object_coords = SkyCoord(
            ra=guide_objects['ra'] * units.deg,
            dec=guide_objects['dec'] * units.deg,
            frame='icrs',
            distance=Distance(parallax=Angle(guide_objects['parallax'], unit=units.mas)),
            pm_ra_cosdec=guide_objects['pmRa'] * units.mas / units.yr,
            pm_dec=guide_objects['pmDec'] * units.mas / units.yr,
            obstime=Time(guide_objects['epoch'], format='jyear_str', scale='tcb')
        )
        object_coords_applied = object_coords.apply_space_motion(new_obstime=_obstime)  # of date
        guide_objects['ra'] = object_coords_applied.ra.deg
        guide_objects['dec'] = object_coords_applied.dec.deg

        return guide_objects, ra, dec, inst_pa

    @staticmethod
    def to_design_id(design_path):

        filename = os.path.splitext(os.path.basename(design_path))[0]
        return int(filename[10:], 0) if filename.startswith('pfsDesign-') else 0

    @staticmethod
    def to_design_path(design_id, design_path=''):

        return os.path.join(design_path, 'pfsDesign-0x{:016x}.fits'.format(design_id))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--magnitude', type=float, default=20.0, help='magnitude limit')
    parser.add_argument('obstime', nargs='?', default=None, help='time of observation (datetime)')
    args, _ = parser.parse_known_args()

    design_id = args.design_id
    design_path = '.' if args.design_id is not None and args.design_path is None else args.design_path
    magnitude = args.magnitude
    obstime = args.obstime
    print('design_id={},design_path={},magnitude={},obstime={}'.format(design_id, design_path, magnitude, obstime))

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='pfs_design')
    guide_objects, ra, dec, inst_pa = pfsDesign(design_id, design_path, logger=logger).get_guide_objects(
        taken_at=obstime
    )
    print('guide_objects={},ra={},dec={},inst_pa={}'.format(guide_objects, ra, dec, inst_pa))
