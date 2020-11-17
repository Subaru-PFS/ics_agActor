import datetime
import os
import sqlite3


_DATABASE = os.path.expandvars('$ICS_MHS_DATA_ROOT/opdb/agc.db')

_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f%z'


def _connect(database=_DATABASE):

    return sqlite3.connect(database)


def insert_target(target_id, ra, dec, pa):

    with _connect() as c:
        c.execute('INSERT INTO target (target_id,ra,dec,pa) VALUES (?,?,?,?)', (target_id, ra, dec, pa))
        c.commit()


def query_target(target_id):

    with _connect() as c:
        return c.execute('SELECT ra,dec,pa FROM target WHERE target_id=?', (target_id, )).fetchone()


def insert_guide_star(target_id, data):

    with _connect() as c:
        c.executemany('INSERT INTO guide_star (target_id,source_id,ra,dec,mag) VALUES (?,?,?,?,?)', ((target_id, *row) for row in data))
        c.commit()


def query_guide_star(target_id):

    with _connect() as c:
        return c.execute('SELECT source_id,ra,dec,mag FROM guide_star WHERE target_id=?', (target_id, )).fetchall()


def insert_agc_exposure(frame_id, pfs_visit_id, exptime, taken_at, azimuth, altitude, inr, adc, inside_temperature, inside_humidity, inside_pressure, outside_temperature, outside_humidity, outside_pressure):

    with _connect() as c:
        c.execute('INSERT INTO agc_exposure (frame_id,pfs_visit_id,exptime,taken_at,azimuth,altitude,inr,adc,inside_temperature,inside_humidity,inside_pressure,outside_temperature,outside_humidity,outside_pressure) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (frame_id, pfs_visit_id, exptime, taken_at.astimezone(tz=datetime.timezone.utc).strftime(_DATETIME_FORMAT), azimuth, altitude, inr, adc, inside_temperature, inside_humidity, inside_pressure, outside_temperature, outside_humidity, outside_pressure))
        c.commit()


def query_agc_exposure(frame_id):

    with _connect() as c:
        data = c.execute('SELECT pfs_visit_id,exptime,taken_at,azimuth,altitude,inr,adc,inside_temperature,inside_humidity,inside_pressure,outside_temperature,outside_humidity,outside_pressure FROM agc_exposure WHERE frame_id=?', (frame_id, )).fetchone()
        return (*data[:2], datetime.datetime.strptime(data[2], _DATETIME_FORMAT), *data[3:])


def insert_agc_data(frame_id, data):

    with _connect() as c:
        c.executemany('INSERT INTO agc_data (frame_id,camera_id,spot_id,image_moment_00,centroid_x,centroid_y,central_image_moment_11,central_image_moment_20,central_image_moment_02,peak_pixel_x,peak_pixel_y,peak_intensity,background) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', ((frame_id, *row) for row in data))
        c.commit()


def query_agc_data(frame_id):

    with _connect() as c:
        return c.execute('SELECT camera_id,spot_id,image_moment_00,centroid_x,centroid_y,central_image_moment_11,central_image_moment_20,central_image_moment_02,peak_pixel_x,peak_pixel_y,peak_intensity,background FROM agc_data WHERE frame_id=?', (frame_id, )).fetchall()


# spt_operational_database-like method names
get_target = query_target
get_guide_star = query_guide_star
get_agc_exposure = query_agc_exposure
get_agc_data = query_agc_data
