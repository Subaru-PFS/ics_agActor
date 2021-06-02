import pfs.utils.opdb


class opDB(pfs.utils.opdb.opDB):

    # override hostname
    pfs.utils.opdb.opDB.host = 'localhost'

    @staticmethod
    def _fetchall(stmt, params=None):

        with opDB.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(stmt, params)
                return curs.fetchall()

    @staticmethod
    def _fetchone(stmt, params=None):

        with opDB.connect() as conn:
            with conn.cursor() as curs:
                curs.execute(stmt, params)
                return curs.fetchone()

    @staticmethod
    def query_tile(tile_id):

        return opDB._fetchone(
            'SELECT ra_center,dec_center,pa FROM tile WHERE tile_id=%s', (tile_id,)
        )

    @staticmethod
    def query_guide_object(tile_id):

        return opDB._fetchall(
            'SELECT source_id,ra,decl,mag FROM guide_object WHERE tile_id=%s', (tile_id,)
        )

    query_guide_star = query_guide_object

    @staticmethod
    def query_agc_exposure(agc_exposure_id):

        return opDB._fetchone(
            'SELECT pfs_visit_id,agc_exptime,taken_at,azimuth,altitude,insrot,adc_pa,outside_temperature,outside_humidity,outside_pressure,m2_pos3 FROM agc_exposure WHERE agc_exposure_id=%s', (agc_exposure_id,)
        )

    @staticmethod
    def query_agc_data(agc_exposure_id):

        return opDB._fetchall(
            'SELECT agc_camera_id,spot_id,image_moment_00_pix,centroid_x_pix,centroid_y_pix,central_image_moment_11_pix,central_image_moment_20_pix,central_image_moment_02_pix,peak_pixel_x_pix,peak_pixel_y_pix,peak_intensity,background,flags FROM agc_data WHERE agc_exposure_id=%s', (agc_exposure_id,)
        )

    @staticmethod
    def insert_tile(tile_id, ra_center, dec_center, pa):

        params = locals().copy()
        params.update(
            program_id=tile_id,
            tile=tile_id,
            is_finished=False
        )
        opDB.insert('tile', **params)

    @staticmethod
    def insert_guide_object(tile_id, data):

        for x in data:
            params = dict(
                tile_id=tile_id,
                source_id=int(x[0]),
                ra=float(x[1]),
                decl=float(x[2]),
                mag=float(x[3])
            )
            opDB.insert('guide_object', **params)

    insert_guide_star = insert_guide_object

    @staticmethod
    def insert_agc_exposure(
            agc_exposure_id,
            pfs_visit_id,
            agc_exptime,
            taken_at,
            azimuth,
            altitude,
            insrot,
            adc_pa,
            outside_temperature,
            outside_humidity,
            outside_pressure,
            m2_pos3
    ):

        params = locals()
        opDB.insert('agc_exposure', **params)

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
