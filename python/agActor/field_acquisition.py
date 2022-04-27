from datetime import datetime, timezone
from numbers import Number
import numpy
#import _gen2_gaia as gaia
import _gen2_gaia_annulus as gaia
import coordinates
from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design
import to_altaz
import kawanomoto


def _parse_kwargs(kwargs):

    if (center := kwargs.pop('center', None)) is not None:
        ra, dec, *optional = center
        kwargs.setdefault('ra', ra)
        kwargs.setdefault('dec', dec)
        if len(optional) > 0:
            kwargs.setdefault('inst_pa', optional[0])
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


def acquire_field(*, frame_id, obswl=0.62, altazimuth=False, logger=None, **kwargs):

    logger and logger.info('frame_id={}'.format(frame_id))
    _parse_kwargs(kwargs)
    taken_at = kwargs.get('taken_at')
    inr = kwargs.get('inr')
    adc = kwargs.get('adc')
    m2_pos3 = kwargs.get('m2_pos3')
    if any(x is None for x in (taken_at, inr, adc, m2_pos3)):
        visit_id, _, _taken_at, _, _, _inr, _adc, _, _, _, _m2_pos3 = opdb.query_agc_exposure(frame_id)
        if (sequence_id := kwargs.get('sequence_id')) is not None:
            # use visit_id from agc_exposure table
            _, _, _inr, _adc, _m2_pos3, _, _, _, _, _taken_at = opdb.query_tel_status(visit_id, sequence_id)
        if taken_at is None: taken_at = _taken_at
        if inr is None: inr = _inr
        if adc is None: adc = _adc
        if m2_pos3 is None: m2_pos3 = _m2_pos3
    logger and logger.info('taken_at={},inr={},adc={},m2_pos3={}'.format(taken_at, inr, adc, m2_pos3))
    detected_objects = opdb.query_agc_data(frame_id)
    #logger and logger.info('detected_objects={}'.format(detected_objects))
    design_id = kwargs.get('design_id')
    design_path = kwargs.get('design_path')
    logger and logger.info('design_id={},design_path={}'.format(design_id, design_path))
    ra = kwargs.get('ra')
    dec = kwargs.get('dec')
    inst_pa = kwargs.get('inst_pa')
    if all(x is None for x in (design_id, design_path)):
        magnitude = kwargs.get('magnitude', 20.0)
        guide_objects, *_ = gaia.get_objects(ra=ra, dec=dec, obstime=taken_at, inr=inr, adc=adc, m2pos3=m2_pos3, obswl=obswl, magnitude=magnitude)
    else:
        if design_path is not None:
            guide_objects, _ra, _dec, _inst_pa = pfs_design(design_id, design_path).guide_stars
        else:
            _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
            guide_objects = opdb.query_pfs_design_agc(design_id)
        if ra is None: ra = _ra
        if dec is None: dec = _dec
        if inst_pa is None: inst_pa = _inst_pa
    logger and logger.info('ra={},dec={},inst_pa={}'.format(ra, dec, inst_pa))
    #logger and logger.info('guide_objects={}'.format(guide_objects))
    return (ra, dec, inst_pa, *_acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=altazimuth, logger=logger))


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
    parser.add_argument('--center', default=None, help='field center coordinates ra, dec[, pa] (deg)')
    parser.add_argument('--magnitude', type=float, default=20.0, help='magnitude limit')
    args, _ = parser.parse_known_args()

    if not (args.center is not None) ^ any(x is not None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --center, --design-id, --design-path')
    center = design = None
    if args.center is not None:
        center = tuple([float(x) for x in args.center.split(',')])
    else:
        design = (args.design_id, args.design_path)
    print('center={},design={}'.format(center, design))

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    ra, dec, inst_pa, dra, ddec, dinr, *values = acquire_field(design=design, frame_id=args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, logger=logger, center=center, magnitude=args.magnitude)
    print('ra={},dec={},inst_pa={},dra={},ddec={},dinr={}'.format(ra, dec, inst_pa, dra, ddec, dinr))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    #print('guide_objects={}'.format(guide_objects))
    #print('detected_objects={}'.format(detected_objects))
    #print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
