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
    def query_agc_exposure(frame_id):

        return opDB._fetchone(
            'SELECT pfs_visit_id,exptime,taken_at,azimuth,altitude,inr,adc,outside_temperature,outside_humidity,outside_pressure,m2_pos3 FROM agc_exposure WHERE frame_id=%s', (frame_id,)
        )

    @staticmethod
    def query_agc_data(frame_id):

        return opDB._fetchall(
            'SELECT camera_id,spot_id,image_moment_00,centroid_x,centroid_y,central_image_moment_11,central_image_moment_20,central_image_moment_02,peak_pixel_x,peak_pixel_y,peak_intensity,background,stepped FROM agc_data WHERE frame_id=%s', (frame_id,)
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
            frame_id,
            pfs_visit_id,
            exptime,
            taken_at,
            azimuth,
            altitude,
            inr,
            adc,
            outside_temperature,
            outside_humidity,
            outside_pressure,
            m2_pos3
    ):

        params = locals()
        opDB.insert('agc_exposure', **params)

    @staticmethod
    def insert_agc_data(frame_id, data):

        for x in data:
            params = dict(
                frame_id=frame_id,
                camera_id=int(x[0]),
                spot_id=int(x[1]),
                image_moment_00=float(x[2]),
                centroid_x=float(x[3]),
                centroid_y=float(x[4]),
                central_image_moment_11=float(x[5]),
                central_image_moment_20=float(x[6]),
                central_image_moment_02=float(x[7]),
                peak_pixel_x=int(x[8]),
                peak_pixel_y=int(x[9]),
                peak_intensity=int(x[10]),
                background=float(x[11]),
                stepped=bool(x[12])
            )
            opDB.insert('agc_data', **params)
