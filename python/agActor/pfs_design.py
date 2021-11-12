import os
import fitsio


class pfsDesign:

    def __init__(self, design_id=None, path=None):

        if design_id is None:
            if path is None:
                raise TypeError('__init__() missing argument(s): \'design_id\' and/or \'path\'')
            elif not os.path.isfile(path):
                raise ValueError('__init__() \'path\' not an existing regular file: \'{}\''.format(path))
        else:
            if path is None:
                path = '/data/pfsDesign'
            elif not os.path.isdir(path):
                raise ValueError('__init__() \'path\' not an existing directory: \'{}\''.format(path))
            path = self.to_design_path(design_id, path)
        self.path = path

    @property
    def guide_stars(self):

        with fitsio.FITS(self.path) as fits:

            header = fits[0].read_header()
            ra = header['RA']
            dec = header['DEC']
            pa = header['POSANG']
            _guide_stars = fits['guidestars'].read()

        #guide_stars = tuple(map(tuple, _guide_stars[['objId', 'ra', 'dec', 'magnitude', 'agId', 'agX', 'agY']]))
        guide_stars = _guide_stars[['objId', 'ra', 'dec', 'magnitude', 'agId', 'agX', 'agY']]
        #guide_stars.dtype.names = ('source_id', 'ra', 'dec', 'mag', 'camera_id', 'x', 'y')

        return guide_stars, ra, dec, pa

    @staticmethod
    def to_design_id(design_path):

        filename = os.path.splitext(os.path.basename(design_path))[0]
        return int(filename[10:], 0) if filename.startswith('pfsDesign-') else 0

    @staticmethod
    def to_design_path(design_id, path=''):

        return os.path.join(path, 'pfsDesign-0x{:016x}.fits'.format(design_id))


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    args, _ = parser.parse_known_args()

    design_id = args.design_id
    design_path = '.' if args.design_id is not None and args.design_path is None else args.design_path
    print('design_id={},design_path={}'.format(design_id, design_path))

    guide_stars, ra, dec, posang = pfsDesign(design_id, design_path).guide_stars
    print('guide_stars={},ra={},dec={},posang={}'.format(guide_stars, ra, dec, posang))
