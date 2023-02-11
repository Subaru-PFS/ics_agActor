import numpy
from opdb import opDB as opdb


def write_agc_guide_offset(*, frame_id, ra=None, dec=None, pa=None, delta_ra=None, delta_dec=None, delta_insrot=None, delta_az=None, delta_el=None, delta_z=None, delta_zs=None):

    params = dict(
        guide_ra=ra,
        guide_dec=dec,
        guide_pa=pa,
        guide_delta_ra=delta_ra,
        guide_delta_dec=delta_dec,
        guide_delta_insrot=delta_insrot,
        guide_delta_az=delta_az,
        guide_delta_el=delta_el,
        guide_delta_z=delta_z
    )
    if delta_zs is not None:
        params.update(guide_delta_z1=delta_zs[0])
        params.update(guide_delta_z2=delta_zs[1])
        params.update(guide_delta_z3=delta_zs[2])
        params.update(guide_delta_z4=delta_zs[3])
        params.update(guide_delta_z5=delta_zs[4])
        params.update(guide_delta_z6=delta_zs[5])
    opdb.insert_agc_guide_offset(frame_id, **params)


def write_agc_match(*, design_id, frame_id, guide_objects, detected_objects, identified_objects):

    data = numpy.array(
        [
            (
                detected_objects['camera_id'][x[0]],
                detected_objects['spot_id'][x[0]],
                guide_objects['source_id'][x[1]],
                x[4], - x[5],  # hsc -> pfs focal plane coordinate system
                x[2], - x[3],  # hsc -> pfs focal plane coordinate system
                0  # flags
            )
            for x in identified_objects
        ],
        dtype=[
            ('agc_camera_id', numpy.int32),
            ('spot_id', numpy.int32),
            ('guide_star_id', numpy.int64),
            ('agc_nominal_x_mm', numpy.float32),
            ('agc_nominal_y_mm', numpy.float32),
            ('agc_center_x_mm', numpy.float32),
            ('agc_center_y_mm', numpy.float32),
            ('flags', numpy.int32)
        ]
    )
    #print(data)
    opdb.insert_agc_match(frame_id, design_id, data)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--design-id', type=lambda x: int(x, 0), default=None, help='design identifier')
    parser.add_argument('--design-path', default=None, help='design path')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    args, _ = parser.parse_known_args()

    if all(x is None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --design-id, --design-path')

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='data_utils')

    import field_acquisition
    import focus

    ra, dec, pa, dra, ddec, dinr, _, dalt, daz, *values = field_acquisition.acquire_field(design=(args.design_id, args.design_path), frame_id=args.frame_id, obswl=args.obswl, altazimuth=True, logger=logger)
    print('ra={},dec={},pa={},dra={},ddec={},dinr={},dalt={},daz={}'.format(ra, dec, pa, dra, ddec, dinr, dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    print('guide_objects={}'.format(guide_objects))
    print('detected_objects={}'.format(detected_objects))
    print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
    dz, dzs = focus._focus(detected_objects=detected_objects, logger=logger)
    print('dz={},dzs={}'.format(dz, dzs))
    write_agc_guide_offset(frame_id=args.frame_id, ra=ra, dec=dec, pa=pa, delta_ra=dra, delta_dec=ddec, delta_insrot=dinr, delta_az=daz, delta_el=dalt, delta_z=dz, delta_zs=dzs)
    write_agc_match(design_id=args.design_id, frame_id=args.frame_id, guide_objects=guide_objects, detected_objects=detected_objects, identified_objects=identified_objects)
