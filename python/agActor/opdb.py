import psycopg2


class opDB:

    @staticmethod
    def _connect():

        return psycopg2.connect(dbname='opdb', host='localhost', user='pfs')

    @staticmethod
    def fetchall(statement, params=None):

        with opDB._connect() as conn:
            with conn.cursor() as curs:
                curs.execute(statement, params)
                return curs.fetchall()

    @staticmethod
    def fetchone(statement, params=None):

        with opDB._connect() as conn:
            with conn.cursor() as curs:
                curs.execute(statement, params)
                return curs.fetchone()

    @staticmethod
    def execute(statement, params=None):

        with opDB._connect() as conn:
            with conn.cursor() as curs:
                curs.execute(statement, params)
            conn.commit()

    @staticmethod
    def insert(table, **params):

        columns = ','.join(params)
        values = ','.join(['%({})s'.format(x) for x in params])
        statement = 'INSERT INTO {} ({}) VALUES ({})'.format(table, columns, values)
        opDB.execute(statement, params)

    @staticmethod
    def query_pfs_design(pfs_design_id):

        return opDB.fetchone(
            'SET TIME ZONE \'UTC\';SELECT tile_id,ra_center_designed,dec_center_designed,pa_designed,num_sci_designed,num_cal_designed,num_sky_designed,num_guide_stars,exptime_tot,exptime_min,ets_version,ets_assigner,designed_at AT TIME ZONE \'HST\',to_be_observed_at AT TIME ZONE \'HST\',is_obsolete FROM pfs_design WHERE pfs_design_id=%s',
            (pfs_design_id,)
        )

    @staticmethod
    def query_tile(tile_id):

        return opDB.fetchone(
            'SELECT ra_center,dec_center,pa FROM tile WHERE tile_id=%s',
            (tile_id,)
        )

    @staticmethod
    def query_pfs_design_agc(pfs_design_id):

        # return opDB.fetchall(
        #     'SELECT guide_star_id,epoch,guide_star_ra,guide_star_dec,guide_star_pm_ra,guide_star_pm_dec,guide_star_parallax,guide_star_magnitude,passband,guide_star_color,agc_camera_id,agc_target_x_pix,agc_target_y_pix,comments FROM pfs_design_agc WHERE pfs_design_id=%s',
        #     (pfs_design_id,)
        # )
        return opDB.fetchall(
            'SELECT guide_star_id,guide_star_ra,guide_star_dec,guide_star_magnitude,agc_camera_id,agc_target_x_pix,agc_target_y_pix FROM pfs_design_agc WHERE pfs_design_id=%s ORDER BY guide_star_id',
            (pfs_design_id,)
        )

    @staticmethod
    def query_agc_exposure(agc_exposure_id):

        return opDB.fetchone(
            'SET TIME ZONE \'UTC\';SELECT pfs_visit_id,agc_exptime,taken_at AT TIME ZONE \'HST\',azimuth,altitude,insrot,adc_pa,outside_temperature,outside_humidity,outside_pressure,m2_pos3 FROM agc_exposure WHERE agc_exposure_id=%s',
            (agc_exposure_id,)
        )

    @staticmethod
    def query_tel_status(pfs_visit_id, status_sequence_id):

        return opDB.fetchone(
            'SET TIME ZONE \'UTC\';SELECT altitude,azimuth,insrot,adc_pa,m2_pos3,tel_ra,tel_dec,dome_shutter_status,dome_light_status,created_at AT TIME ZONE \'HST\' FROM tel_status WHERE pfs_visit_id=%s AND status_sequence_id=%s',
            (pfs_visit_id, status_sequence_id,)
        )

    @staticmethod
    def query_agc_data(agc_exposure_id):

        return opDB.fetchall(
            'SELECT agc_camera_id,spot_id,image_moment_00_pix,centroid_x_pix,centroid_y_pix,central_image_moment_11_pix,central_image_moment_20_pix,central_image_moment_02_pix,peak_pixel_x_pix,peak_pixel_y_pix,peak_intensity,background,COALESCE(flags,CAST(centroid_x_pix>=511.5+24 AS INTEGER)) AS flags FROM agc_data WHERE agc_exposure_id=%s ORDER BY agc_camera_id,spot_id',
            (agc_exposure_id,)
        )

    @staticmethod
    def query_agc_guide_offset(agc_exposure_id):

        return opDB.fetchone(
            'SELECT guide_ra,guide_dec,guide_pa,guide_delta_ra,guide_delta_dec,guide_delta_insrot,guide_delta_az,guide_delta_el,guide_delta_z,guide_delta_z1,guide_delta_z2,guide_delta_z3,guide_delta_z4,guide_delta_z5,guide_delta_z6 FROM agc_guide_offset WHERE agc_exposure_id=%s',
            (agc_exposure_id,)
        )

    @staticmethod
    def query_agc_match(agc_exposure_id):

        return opDB.fetchall(
            'SELECT agc_camera_id,spot_id,pfs_design_id,guide_star_id,agc_nominal_x_mm,agc_nominal_y_mm,agc_center_x_mm,agc_center_y_mm,flags FROM agc_match WHERE agc_exposure_id=%s ORDER BY agc_camera_id,spot_id',
            (agc_exposure_id,)
        )

    @staticmethod
    def insert_pfs_design(pfs_design_id, tile_id, **params):

        params.update(pfs_design_id=pfs_design_id, tile_id=tile_id)
        opDB.insert('pfs_design', **params)

    @staticmethod
    def insert_tile(**params):

        columns = ','.join(params)
        values = ','.join(['%({})s'.format(x) for x in params])
        statement = 'INSERT INTO tile ({}) VALUES ({}) RETURNING tile_id'.format(columns, values)
        return opDB.fetchone(statement, params)[0]

    @staticmethod
    def insert_pfs_design_agc(pfs_design_id, data):

        for x in data:
            params = dict(
                pfs_design_id=pfs_design_id,
                guide_star_id=int(x[0]),
                guide_star_ra=float(x[1]),
                guide_star_dec=float(x[2]),
                guide_star_magnitude=float(x[3]),
                agc_camera_id=int(x[4]),
                agc_target_x_pix=float(x[5]),
                agc_target_y_pix=float(x[6])
            )
            opDB.insert('pfs_design_agc', **params)

    @staticmethod
    def insert_agc_exposure(agc_exposure_id, **params):

        params.update(agc_exposure_id=agc_exposure_id)
        opDB.insert('agc_exposure', **params)

    @staticmethod
    def insert_tel_status(pfs_visit_id, status_sequence_id, **params):

        params.update(pfs_visit_id=pfs_visit_id, status_sequence_id=status_sequence_id)
        opDB.insert('tel_status', **params)

    @staticmethod
    def insert_agc_data(agc_exposure_id, data):

        for x in data:
            params = dict(
                agc_exposure_id=agc_exposure_id,
                agc_camera_id=int(x[0]),
                spot_id=int(x[1]),
                image_moment_00_pix=float(x[2]),
                centroid_x_pix=float(x[3]),
                centroid_y_pix=float(x[4]),
                central_image_moment_11_pix=float(x[5]),
                central_image_moment_20_pix=float(x[6]),
                central_image_moment_02_pix=float(x[7]),
                peak_pixel_x_pix=int(x[8]),
                peak_pixel_y_pix=int(x[9]),
                peak_intensity=float(x[10]),
                background=float(x[11]),
                flags=int(x[12])
            )
            opDB.insert('agc_data', **params)

    @staticmethod
    def insert_agc_guide_offset(agc_exposure_id, **params):

        params.update(agc_exposure_id=agc_exposure_id)
        opDB.insert('agc_guide_offset', **params)

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
                flags=int(x[7])
            )
            opDB.insert('agc_match', **params)

    @staticmethod
    def update_pfs_design(pfs_design_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE pfs_design SET {} WHERE pfs_design_id=%(pfs_design_id)s'.format(column_values)
        params.update(pfs_design_id=pfs_design_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_tile(tile_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE tile SET {} WHERE tile_id=%(tile_id)s'.format(column_values)
        params.update(tile_id=tile_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_pfs_design_agc(pfs_design_id, guide_star_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE pfs_design_agc SET {} WHERE pfs_design_id=%(pfs_design_id)s AND guide_star_id=%(guide_star_id)s'.format(column_values)
        params.update(pfs_design_id=pfs_design_id, guide_star_id=guide_star_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_agc_exposure(agc_exposure_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE agc_exposure SET {} WHERE agc_exposure_id=%(agc_exposure_id)s'.format(column_values)
        params.update(agc_exposure_id=agc_exposure_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_tel_status(pfs_visit_id, status_sequence_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE tel_status SET {} WHERE pfs_visit_id=%(pfs_visit_id)s AND status_sequence_id=%(status_sequence_id)s'.format(column_values)
        params.update(pfs_visit_id=pfs_visit_id, status_sequence_id=status_sequence_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_agc_data(agc_exposure_id, agc_camera_id, spot_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE agc_data SET {} WHERE agc_exposure_id=%(agc_exposure_id)s AND agc_camera_id=%(agc_camera_id)s AND spot_id=%(spot_id)s'.format(column_values)
        params.update(agc_exposure_id=agc_exposure_id, agc_camera_id=agc_camera_id, spot_id=spot_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_agc_guide_offset(agc_exposure_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE agc_guide_offset SET {} WHERE agc_exposure_id=%(agc_exposure_id)s'.format(column_values)
        params.update(agc_exposure_id=agc_exposure_id)
        opDB.execute(statement, params)

    @staticmethod
    def update_agc_match(agc_exposure_id, agc_camera_id, spot_id, **params):

        column_values = ','.join(['{}=%({})s'.format(x, x) for x in params])
        statement = 'UPDATE agc_match SET {} WHERE agc_exposure_id=%(agc_exposure_id)s AND agc_camera_id=%(agc_camera_id)s AND spot_id=%(spot_id)s'.format(column_values)
        params.update(agc_exposure_id=agc_exposure_id, agc_camera_id=agc_camera_id, spot_id=spot_id)
        opDB.execute(statement, params)
