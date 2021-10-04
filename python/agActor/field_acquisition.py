import numpy
import coordinates
from opdb import opDB as opdb
import to_altaz
import kawanomoto


def acquire_field(design_id, frame_id, obswl=0.62, altazimuth=False, verbose=False, logger=None):

    _, ra, dec, *_ = opdb.query_pfs_design(design_id)
    logger and logger.info('ra={},dec={}'.format(ra, dec))

    guide_objects = opdb.query_pfs_design_agc(design_id)

    _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))

    detected_objects = opdb.query_agc_data(frame_id)

    return _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=altazimuth, verbose=verbose, logger=logger)


def _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=5.5, obswl=0.62, altazimuth=False, verbose=False, logger=None):

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = numpy.sqrt(numpy.square((x2 - y2) / 2) + numpy.square(xy))
        a = numpy.sqrt(p + q)
        b = numpy.sqrt(p - q)
        return a, b

    _guide_objects = numpy.array([(x[1], x[2], x[3]) for x in guide_objects])

    _detected_objects = numpy.array([
        (
            x[0],
            x[1],
            *coordinates.det2dp(int(x[0]), x[3], x[4]),
            x[10],
            *semi_axes(x[5], x[6], x[7]),
            x[-1]
        ) for x in detected_objects
    ])

    pfs = kawanomoto.FieldAcquisitionAndFocusing.PFS()
    dra, ddec, dinr, *diags = pfs.FA(_guide_objects, _detected_objects, ra, dec, taken_at, adc, inr, m2_pos3, obswl)
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))

    values = ()

    if altazimuth:

        _, _, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
        logger and logger.info('dalt={},daz={}'.format(dalt, daz))

        values = dalt, daz

    if verbose:

        v, f, min_dist_index_f, obj_x, obj_y, cat_x, cat_y = diags
        index_v, = numpy.where(v)
        index_f, = numpy.where(f)
        identified_objects = [
            (
                k,  # index of detected object
                int(x[0]),  # index of identified guide object
                float(x[1]), float(x[2]),  # detector plane coordinates of detected object
                float(x[3]), float(x[4]),  # detector plane coordinates of identified guide object
                *coordinates.dp2det(detected_objects[k][0], float(x[3]), float(x[4]))  # detector coordinates of identified guide object
            ) for k, x in ((int(index_v[int(index_f[i])]), x) for i, x in enumerate(zip(min_dist_index_f, obj_x, obj_y, cat_x, cat_y)))
        ]

        # find "representative" spot size, peak intensity, and flux by "median" of pointing errors
        dx = 0  # mm
        dy = 0  # mm (HSC definition)
        size = 0  # pix
        peak = 0  # pix
        flux = 0  # pix
        esq = [(x[2] - x[4]) ** 2 + (x[3] - x[5]) ** 2 for x in identified_objects]  # squares of pointing errors in detector plane coordinates
        n = len(esq) - numpy.isnan(esq).sum()
        if n > 0:
            i = numpy.argpartition(esq, n // 2)[n // 2]  # index of "median" of identified objects
            dx = (identified_objects[i][2] - identified_objects[i][4])
            dy = (identified_objects[i][3] - identified_objects[i][5])
            k = identified_objects[i][0]  # index of "median" of detected objects
            a, b = semi_axes(*detected_objects[k][5:8])
            size = (a * b) ** 0.5
            peak = detected_objects[k][10]
            flux = detected_objects[k][2]

        values = *values, guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux

    return (dra, ddec, dinr, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=int, required=True, help='design identifier')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    dra, ddec, dinr, *values = acquire_field(args.design_id, args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, verbose=args.verbose, logger=logger)
    print('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    if args.verbose:
        guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
        print('guide_objects={}'.format(guide_objects))
        print('detected_objects={}'.format(detected_objects))
        print('identified_objects={}'.format(identified_objects))
        print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
