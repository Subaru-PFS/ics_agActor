from datetime import datetime, timezone
from numbers import Number
import numpy
import coordinates
from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design
import to_altaz
import kawanomoto


def acquire_field(*, design=None, frame_id, obswl=0.62, altazimuth=False, logger=None, **kwargs):

    tel_status = kwargs.get('tel_status')
    status_id = kwargs.get('status_id')
    if tel_status is not None:
        _, _, inr, adc, m2_pos3, _, _, _, taken_at = tel_status
    elif status_id is not None:
        # visit_id can be obtained from agc_exposure table
        visit_id, sequence_id = status_id
        _, _, inr, adc, m2_pos3, _, _, _, _, taken_at = opdb.query_tel_status(visit_id, sequence_id)
    else:
        _, _, taken_at, _, _, inr, adc, _, _, _, m2_pos3 = opdb.query_agc_exposure(frame_id)
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))
    detected_objects = opdb.query_agc_data(frame_id)
    design_id, design_path = design
    logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))
    if design_path is not None:
        guide_objects, ra, dec, pa = pfs_design(design_id, design_path).guide_stars
        logger and logger.info('ra={},dec={},pa={}'.format(ra, dec, pa))
    else:
        _, ra, dec, pa, *_ = opdb.query_pfs_design(design_id)
        logger and logger.info('ra={},dec={},pa={}'.format(ra, dec, pa))
        guide_objects = opdb.query_pfs_design_agc(design_id)
    return (ra, dec, pa, *_acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=altazimuth, logger=logger))


def _acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=6.0, obswl=0.62, altazimuth=False, logger=None):

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = numpy.sqrt(numpy.square((x2 - y2) / 2) + numpy.square(xy))
        a = numpy.sqrt(p + q)
        b = numpy.sqrt(p - q)
        return a, b

    _guide_objects = numpy.array([(x[1], x[2], x[3]) for x in guide_objects])
    _detected_objects = numpy.array(
        [
            (
                x[0],
                x[1],
                *coordinates.det2dp(int(x[0]), x[3], x[4]),
                x[10],
                *semi_axes(x[5], x[6], x[7]),
                x[-1]
            )
            for x in detected_objects
        ]
    )
    pfs = kawanomoto.FieldAcquisitionAndFocusing.PFS()
    dra, ddec, dinr, *diags = pfs.FA(_guide_objects, _detected_objects, ra, dec, taken_at.astimezone(tz=timezone.utc) if isinstance(taken_at, datetime) else datetime.fromtimestamp(taken_at, tz=timezone.utc) if isinstance(taken_at, Number) else taken_at, adc, inr, m2_pos3, obswl)
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
    values = ()
    if altazimuth:
        alt, az, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
        logger and logger.info('alt={},az={},dalt={},daz={}'.format(alt, az, dalt, daz))
        values = dalt, daz
    guide_objects = numpy.array(
        [(x[0], x[1], x[2], x[3]) for x in guide_objects],
        dtype=[
            ('source_id', numpy.int64),  # u8 (80) not supported by FITSIO
            ('ra', numpy.float64),
            ('dec', numpy.float64),
            ('mag', numpy.float32)
        ]
    )
    detected_objects = numpy.array(
        detected_objects,
        dtype=[
            ('camera_id', numpy.int16),
            ('spot_id', numpy.int16),
            ('moment_00', numpy.float32),
            ('centroid_x', numpy.float32),
            ('centroid_y', numpy.float32),
            ('central_moment_11', numpy.float32),
            ('central_moment_20', numpy.float32),
            ('central_moment_02', numpy.float32),
            ('peak_x', numpy.uint16),
            ('peak_y', numpy.uint16),
            ('peak', numpy.uint16),
            ('background', numpy.float32),
            ('flags', numpy.uint8)
        ]
    )
    v, f, min_dist_index_f, obj_x, obj_y, cat_x, cat_y = diags
    index_v, = numpy.where(v)
    index_f, = numpy.where(f)
    identified_objects = numpy.array(
        [
            (
                k,  # index of detected object
                int(x[0]),  # index of identified guide object
                float(x[1]), float(x[2]),  # detector plane coordinates of detected object
                float(x[3]), float(x[4]),  # detector plane coordinates of identified guide object
                *coordinates.dp2det(detected_objects[k][0], float(x[3]), float(x[4]))  # detector coordinates of identified guide object
            )
            for k, x in ((int(index_v[int(index_f[i])]), x) for i, x in enumerate(zip(min_dist_index_f, obj_x, obj_y, cat_x, cat_y)))
        ],
        dtype=[
            ('detected_object_id', numpy.int16),
            ('guide_object_id', numpy.int16),
            ('detected_object_x', numpy.float32),
            ('detected_object_y', numpy.float32),
            ('guide_object_x', numpy.float32),
            ('guide_object_y', numpy.float32),
            ('guide_object_xdet', numpy.float32),
            ('guide_object_ydet', numpy.float32)
        ]
    )
    # find "representative" spot size, peak intensity, and flux by "median" of pointing errors
    dx = 0  # mm
    dy = 0  # mm (HSC definition)
    size = 0  # pix
    peak = 0  # pix
    flux = 0  # pix
    esq = (identified_objects['detected_object_x'] - identified_objects['guide_object_x']) ** 2 + (identified_objects['detected_object_y'] - identified_objects['guide_object_y']) ** 2  # squares of pointing errors in detector plane coordinates
    n = len(esq) - numpy.isnan(esq).sum()
    if n > 0:
        i = numpy.argpartition(esq, n // 2)[n // 2]  # index of "median" of identified objects
        dx = identified_objects['detected_object_x'][i] - identified_objects['guide_object_x'][i]
        dy = identified_objects['detected_object_y'][i] - identified_objects['guide_object_y'][i]
        k = identified_objects['detected_object_id'][i]  # index of "median" of detected objects
        a, b = semi_axes(detected_objects['central_moment_11'][k], detected_objects['central_moment_20'][k], detected_objects['central_moment_02'][k])
        size = (a * b) ** 0.5
        peak = detected_objects['peak'][k]
        flux = detected_objects['moment_00'][k]
    values = *values, guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux
    return (dra, ddec, dinr, *values)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    args, _ = parser.parse_known_args()

    if all(x is None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --design-id, --design-path')

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    ra, dec, pa, dra, ddec, dinr, *values = acquire_field(design=(args.design_id, args.design_path), frame_id=args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, logger=logger)
    print('ra={},dec={},pa={},dra={},ddec={},dinr={}'.format(ra, dec, pa, dra, ddec, dinr))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    print('guide_objects={}'.format(guide_objects))
    print('detected_objects={}'.format(detected_objects))
    print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
