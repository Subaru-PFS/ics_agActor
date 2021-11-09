from opdb import opDB as opdb
import _gen2_gaia as gaia
import field_acquisition


def acquire_field(*, ra, dec, frame_id, status_id=None, tel_status=None, obswl=0.62, altazimuth=False, logger=None):

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
    guide_objects, *_ = gaia.get_objects(ra=ra, dec=dec, obstime=taken_at, inr=inr, adc=adc, m2pos3=m2_pos3, obswl=obswl)
    return field_acquisition._acquire_field(guide_objects, detected_objects, ra, dec, taken_at, adc, inr, m2_pos3=m2_pos3, obswl=obswl, altazimuth=altazimuth, logger=logger)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('ra', type=float, help='right ascension (ICRS) of the field center (hr)')
    parser.add_argument('dec', type=float, help='declination (ICRS) of the field center (deg)')
    parser.add_argument('--frame-id', type=int, required=True, help='frame identifier')
    parser.add_argument('--obswl', type=float, default=0.62, help='wavelength of observation (um)')
    parser.add_argument('--altazimuth', action='store_true', help='')
    args, _ = parser.parse_known_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name='field_acquisition')
    dra, ddec, dinr, *values = acquire_field(ra=args.ra, dec=args.dec, frame_id=args.frame_id, obswl=args.obswl, altazimuth=args.altazimuth, logger=logger)
    print('dra={},ddec={},dinr={}'.format(dra, ddec, dinr))
    if args.altazimuth:
        dalt, daz, *values = values
        print('dalt={},daz={}'.format(dalt, daz))
    guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux = values
    print('guide_objects={}'.format(guide_objects))
    print('detected_objects={}'.format(detected_objects))
    print('identified_objects={}'.format(identified_objects))
    print('dx={},dy={},size={},peak={},flux={}'.format(dx, dy, size, peak, flux))
