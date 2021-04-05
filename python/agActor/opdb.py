import datetime
import os
import sqlite3


_DATABASE = os.path.expandvars('$ICS_MHS_DATA_ROOT/opdb/agc.db')

_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f%z'


def _connect(database=_DATABASE):

    return sqlite3.connect(database)


def insert_tile(tile_id, ra_center, dec_center, pa):

    program_id = tile_id
    tile = tile_id
    is_finished = False
    with _connect() as c:
        c.execute('INSERT INTO tile (tile_id,program_id,tile,ra_center,dec_center,pa,is_finished) VALUES (?,?,?,?,?,?,?)', (tile_id, program_id, tile, ra_center, dec_center, pa, int(is_finished)))
        c.commit()


def query_tile(tile_id):

    with _connect() as c:
        return c.execute('SELECT ra_center,dec_center,pa FROM tile WHERE tile_id=?', (tile_id, )).fetchone()


def insert_guide_star(tile_id, data):

    with _connect() as c:
        c.executemany('INSERT INTO guide_star (tile_id,source_id,ra,dec,mag) VALUES (?,?,?,?,?)', ((tile_id, *row) for row in data))
        c.commit()


def query_guide_star(tile_id):

    with _connect() as c:
        return c.execute('SELECT source_id,ra,dec,mag FROM guide_star WHERE tile_id=?', (tile_id, )).fetchall()


def insert_agc_exposure(frame_id, pfs_visit_id, exptime, taken_at, azimuth, altitude, inr, adc, inside_temperature, inside_humidity, inside_pressure, outside_temperature, outside_humidity, outside_pressure, m2_pos3):

    with _connect() as c:
        c.execute('INSERT INTO agc_exposure (frame_id,pfs_visit_id,exptime,taken_at,azimuth,altitude,inr,adc,inside_temperature,inside_humidity,inside_pressure,outside_temperature,outside_humidity,outside_pressure,m2_pos3) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (frame_id, pfs_visit_id, exptime, taken_at.astimezone(tz=datetime.timezone.utc).strftime(_DATETIME_FORMAT), azimuth, altitude, inr, adc, inside_temperature, inside_humidity, inside_pressure, outside_temperature, outside_humidity, outside_pressure, m2_pos3))
        c.commit()


def query_agc_exposure(frame_id):

    with _connect() as c:
        data = c.execute('SELECT pfs_visit_id,exptime,taken_at,azimuth,altitude,inr,adc,inside_temperature,inside_humidity,inside_pressure,outside_temperature,outside_humidity,outside_pressure,m2_pos3 FROM agc_exposure WHERE frame_id=?', (frame_id, )).fetchone()
        m2_pos3 = data[-1]
        if m2_pos3 is None:
            m2_pos3 = 5.5   # nominally 5-6 mm for HSC
        return (*data[:2], datetime.datetime.strptime(data[2], _DATETIME_FORMAT), *data[3:-1], m2_pos3)


def insert_agc_data(frame_id, data):

    with _connect() as c:
        c.executemany('INSERT INTO agc_data (frame_id,camera_id,spot_id,image_moment_00,centroid_x,centroid_y,central_image_moment_11,central_image_moment_20,central_image_moment_02,peak_pixel_x,peak_pixel_y,peak_intensity,background,stepped) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', ((frame_id, *row) for row in data))
        c.commit()


def query_agc_data(frame_id):

    with _connect() as c:
        data = c.execute('SELECT camera_id,spot_id,image_moment_00,centroid_x,centroid_y,central_image_moment_11,central_image_moment_20,central_image_moment_02,peak_pixel_x,peak_pixel_y,peak_intensity,background,stepped FROM agc_data WHERE frame_id=?', (frame_id, )).fetchall()
    return [(*x[:-1], int(x[3] >= 511.5)) if x[-1] is None else x for x in data]


# spt_operational_database-like method names
get_tile = query_tile
get_guide_star = query_guide_star
get_agc_exposure = query_agc_exposure
get_agc_data = query_agc_data
