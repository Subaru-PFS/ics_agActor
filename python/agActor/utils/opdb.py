from ics.utils.opdb import opDB as icsDB


class opDB(icsDB):
    @staticmethod
    def query_pfs_design(pfs_design_id):
        sql = """
SET TIME ZONE 'UTC';
SELECT
    tile_id,
    ra_center_designed,
    dec_center_designed,
    pa_designed,
    num_sci_designed,
    num_cal_designed,
    num_sky_designed,
    num_guide_stars,
    exptime_tot,
    exptime_min,
    ets_version,
    ets_assigner,
    designed_at AT TIME ZONE 'HST',
    to_be_observed_at AT TIME ZONE 'HST',
    is_obsolete
FROM pfs_design
WHERE pfs_design_id=%s
"""
        return opDB.fetchone(sql, (pfs_design_id,))

    @staticmethod
    def query_pfs_design_agc(pfs_design_id):
        sql = """
SELECT
    guide_star_id,
    guide_star_ra,
    guide_star_dec,
    guide_star_magnitude,
    agc_camera_id,
    agc_target_x_pix,
    agc_target_y_pix,
    guide_star_flag
FROM pfs_design_agc
WHERE pfs_design_id=%s
ORDER BY guide_star_id
"""
        return opDB.fetchall(sql, (pfs_design_id,))

    @staticmethod
    def query_agc_exposure(agc_exposure_id):
        sql = """
SET TIME ZONE 'UTC';
SELECT
    pfs_visit_id,
    agc_exptime,
    taken_at AT TIME ZONE 'HST',
    azimuth,
    altitude,
    insrot,
    adc_pa,
    outside_temperature,
    outside_humidity,
    outside_pressure,
    m2_pos3
FROM agc_exposure
WHERE agc_exposure_id=%s
"""
        return opDB.fetchone(sql, (agc_exposure_id,))

    @staticmethod
    def query_tel_status(pfs_visit_id, status_sequence_id):
        sql = """
SET TIME ZONE 'UTC';
SELECT
    altitude,
    azimuth,
    insrot,
    adc_pa,
    m2_pos3,
    tel_ra,
    tel_dec,
    dome_shutter_status,
    dome_light_status,
    created_at AT TIME ZONE 'HST'
FROM tel_status
WHERE pfs_visit_id=%s AND status_sequence_id=%s
"""
        return opDB.fetchone(
            sql,
            (
                pfs_visit_id,
                status_sequence_id,
            ),
        )

    @staticmethod
    def query_agc_data(agc_exposure_id):
        sql = """
SELECT agc_camera_id,
     spot_id,
     image_moment_00_pix,
     centroid_x_pix,
     centroid_y_pix,
     central_image_moment_11_pix,
     central_image_moment_20_pix,
     central_image_moment_02_pix,
     peak_pixel_x_pix,
     peak_pixel_y_pix,
     peak_intensity,
     background,
     COALESCE(flags, CAST(centroid_x_pix >= 511.5 + 24 AS INTEGER)) AS flags
FROM agc_data
WHERE agc_exposure_id = %s
ORDER BY agc_camera_id, spot_id
"""
        return opDB.fetchall(sql, (agc_exposure_id,))

    @staticmethod
    def insert_agc_guide_offset(agc_exposure_id, **params):
        params.update(agc_exposure_id=agc_exposure_id)
        opDB.insert("agc_guide_offset", **params)

    @staticmethod
    def insert_agc_match(agc_exposure_id, pfs_design_id, data):
        for x in data:
            params = dict(
                agc_exposure_id=agc_exposure_id,
                agc_camera_id=int(x[0]),
                spot_id=int(x[1]),
                pfs_design_id=pfs_design_id,
                guide_star_id=int(x[2]),
                agc_nominal_x_mm=float(x[3]),
                agc_nominal_y_mm=float(x[4]),
                agc_center_x_mm=float(x[5]),
                agc_center_y_mm=float(x[6]),
                flags=int(x[7]),
            )
            opDB.insert("agc_match", **params)
