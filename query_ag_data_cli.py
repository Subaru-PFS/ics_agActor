#!/usr/bin/env python3
"""
Query AG actor database for a given design_id.

Usage:
    python query_ag_data_cli.py <design_id> [--frames <id> [<id> ...]] [--table <name>]

Tables: exposure_info, detected, matched, offsets  (default: all)

Outputs JSON to stdout.
"""
import argparse
import json
import sys

import pandas as pd
from pfs.utils.database.opdb import OpDB


def df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-serialisable list of dicts."""
    return json.loads(df.to_json(orient="records"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design_id", type=int, help="pfs_design_id integer")
    parser.add_argument("--frames", type=int, nargs="+",
                        help="Restrict to specific agc_exposure_ids (last 2 digits of frame number ok if unambiguous)")
    parser.add_argument("--table", choices=["exposure_info", "detected", "matched", "offsets"],
                        default=None, help="Only output one table (default: all)")
    args = parser.parse_args()

    OpDB.set_default_connection(host="pfsa-db", user="public_user")
    opdb = OpDB()

    # --- Exposure info ---
    exposure_info = opdb.query_dataframe("""
        SELECT t1.*
        FROM pfs_visit t0, agc_exposure t1
        WHERE t0.pfs_design_id=:design_id
          AND t0.pfs_visit_id=t1.pfs_visit_id
        ORDER BY t1.agc_exposure_id
    """, params=dict(design_id=args.design_id)).reset_index(drop=True)

    frame_ids = exposure_info.agc_exposure_id.tolist()

    # Filter to specific frames if requested
    if args.frames:
        requested = set(args.frames)
        # Support both full IDs and short suffixes
        if all(f < 10000 for f in requested):
            frame_ids = [fid for fid in frame_ids if (fid % 10000) in requested]
        else:
            frame_ids = [fid for fid in frame_ids if fid in requested]
        exposure_info = exposure_info[exposure_info.agc_exposure_id.isin(frame_ids)].reset_index(drop=True)

    if not frame_ids:
        print(json.dumps({"error": "No frames found for given design_id / frame filter"}))
        sys.exit(1)

    result = {}

    if args.table in (None, "exposure_info"):
        result["exposure_info"] = df_to_records(exposure_info)

    if args.table in (None, "detected"):
        detected = opdb.query_dataframe("""
            SELECT agc_exposure_id, agc_camera_id, spot_id,
                   image_moment_00_pix,
                   centroid_x_pix, centroid_y_pix,
                   central_image_moment_11_pix,
                   central_image_moment_20_pix,
                   central_image_moment_02_pix,
                   peak_pixel_x_pix, peak_pixel_y_pix,
                   peak_intensity, background,
                   COALESCE(flags, CAST(centroid_x_pix >= 511.5 + 24 AS INTEGER)) AS flags
            FROM agc_data
            WHERE agc_exposure_id = ANY(:frame_ids)
            ORDER BY agc_exposure_id, agc_camera_id, spot_id
        """, params={"frame_ids": frame_ids})
        result["detected"] = df_to_records(detected)

    if args.table in (None, "matched"):
        matched = opdb.query_dataframe(
            "SELECT * FROM agc_match WHERE agc_exposure_id = ANY(:frame_ids) ORDER BY agc_exposure_id, agc_camera_id",
            params={"frame_ids": frame_ids}
        )
        result["matched"] = df_to_records(matched)

    if args.table in (None, "offsets"):
        offsets = opdb.query_dataframe("""
            SELECT * FROM agc_guide_offset
            WHERE agc_exposure_id = ANY(:frame_ids)
            ORDER BY agc_exposure_id
        """, params=dict(frame_ids=frame_ids))
        result["offsets"] = df_to_records(offsets)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
