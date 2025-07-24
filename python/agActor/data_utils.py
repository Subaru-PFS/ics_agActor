import numpy as np

from opdb import opDB as opdb


def write_agc_guide_offset(
    *,
    frame_id,
    ra=None,
    dec=None,
    pa=None,
    delta_ra=None,
    delta_dec=None,
    delta_insrot=None,
    delta_scale=None,
    delta_az=None,
    delta_el=None,
    delta_z=None,
    delta_zs=None,
):

    params = dict(
        guide_ra=ra,
        guide_dec=dec,
        guide_pa=pa,
        guide_delta_ra=delta_ra,
        guide_delta_dec=delta_dec,
        guide_delta_insrot=delta_insrot,
        guide_delta_scale=delta_scale,
        guide_delta_az=delta_az,
        guide_delta_el=delta_el,
        guide_delta_z=delta_z,
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

    data = np.array(
        [
            (
                detected_objects["camera_id"][x[0]],
                detected_objects["spot_id"][x[0]],
                guide_objects["source_id"][x[1]],
                x[4],
                -x[5],  # hsc -> pfs focal plane coordinate system
                x[2],
                -x[3],  # hsc -> pfs focal plane coordinate system
                guide_objects["filter_flag"][x[1]],  # flags
            )
            for x in identified_objects
        ],
        dtype=[
            ("agc_camera_id", np.int32),
            ("spot_id", np.int32),
            ("guide_star_id", np.int64),
            ("agc_nominal_x_mm", np.float32),
            ("agc_nominal_y_mm", np.float32),
            ("agc_center_x_mm", np.float32),
            ("agc_center_y_mm", np.float32),
            ("flags", np.int32),
        ],
    )
    # print(data)
    opdb.insert_agc_match(frame_id, design_id, data)
