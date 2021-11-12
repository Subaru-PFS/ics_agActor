import numpy
from opdb import opDB as opdb


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
    parser.add_argument('--altazimuth', action='store_true', help='')
    args, _ = parser.parse_known_args()

    if all(x is None for x in (args.design_id, args.design_path)):
        parser.error('at least one of the following arguments is required: --design-id, --design-path')

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='data_utils')

    import field_acquisition

    ra, dec, pa, dra, ddec, dinr, *values = field_acquisition.acquire_field(design=(args.design_id, args.design_path), frame_id=args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, logger=logger)
    print('ra={},dec={},pa={},dra={},ddec={},dinr={}'.format(ra, dec, pa, dra, ddec, dinr))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    print('guide_objects={}'.format(guide_objects))
    print('detected_objects={}'.format(detected_objects))
    print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
    write_agc_match(design_id=args.design_id, frame_id=args.frame_id, guide_objects=guide_objects, detected_objects=detected_objects, identified_objects=identified_objects)
