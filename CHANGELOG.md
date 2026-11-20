# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- INSTRM-3114 - ics_agccActor now stores each spot's second moments as its variance; they were half of it. `query_agc_data` returns the variance for any exposure: it doubles the moments of exposures whose `agc_exposure.version_actor` is the literal `"git"`, which is what agcc wrote until it recorded its version, so focus and size are the same for exposures from before and after that agcc release, live or replayed. **Deploy this release before the agcc release, or with it**: it works with either agcc, while the agcc release with an older ics_agActor doubles the slope term of `dZ` and sends a size sqrt(2) too large. The focus slope is 0.0043 mm/pix^2 on the variance, half of 0.0086, so `dZ` is what it was; the intercept (-0.026 mm) is unchanged. The spot size sent to mlp1 (`size=`) is 2.05 times the radius of the variance (`variancePerMoment=1` in `pfs.utils.agMetrics`), the same FWHM as before. The moments in the files `save_numpy_files` writes are the variance too, twice what they were, and `max_size` and `min_size`, where a script passes them, are compared with semi-axes sqrt(2) larger. **Needs the pfs_utils release with `variancePerMoment`, switched in at the same restart**: with an older pfs_utils the size raises a `TypeError` on the first frame, and ics_agActor 1.1.85 with that pfs_utils sends a size sqrt(2) too small.

## [1.1.85] - 2026-11-16

- INSTRM-3069 - The spot size sent to mlp1 (`size=`) is the FWHM from `pfs.utils.agMetrics`: 2.9 times the radius of the stored second moments (trace) at 0.137 arcsec per pixel, the definition drp_qa uses. It was twice the determinant radius at 0.179 arcsec per pixel (a 15 m focal length). The new value is at least 11% larger, more for elongated spots (13% in the median, 0.64 to 0.71 arcsec, over the 290 exposures of ten Run 30 engineering visits), and so is the SEEING card of the FITS headers, which the telescope fills from it. `acquire_field` and the autoguide loop also log, for each exposure, that FWHM for the two halves of the detectors separately, with the number of spots on each. `find_representative_spot` no longer takes `ag_plate_scale`. **Needs pfs_utils 7.4.45 or later.**

## [1.1.84] - 2026-11-16

- INSTRM-2800 - A guide offset is used only when at least `min_matched_stars` (default 3) good matches from enabled cameras made it. With fewer, the guide loop sends nothing, reports `guideErrors` status `TOO_FEW_STARS`, raises the warning alert `AG.TOO_FEW_STARS` and writes `agc_guide_offset.mask = INVALID_OFFSET | TOO_FEW_STARS` (3); `acquire_field` records the offset the same way and, when asked to guide (the default), fails, saying how many stars matched, were detected and were in the catalog; with `guide=no` it warns with the same message and finishes. Either way `guideErrors` then carries NaN for dRA, dDec, dInR, dAz, dAlt and dScale (dZ kept), since Gen2 copies it into `PFS.AG.ERR` and `SETUPFIELD.sk` applies those offsets after acquisition without reading the status; out-of-range offsets keep their values. New `min_matched_stars` option on `acquire_field` and `autoguide start/initialize/reconfigure`. A frame with no real match still gave an offset, from detections paired with the wrong guide stars, and offsets from 0-2 stars were sent in Runs 29 and 30. Focus now says which cameras gave no value and why (a camera needs a star on each half); `ag focus` fails with that instead of `dz=nan`. `GuideOffsetFlag` now comes from pfs_utils (`pfs.utils.datamodel.ag`); `agActor.utils.data.GuideOffsetFlag` still works. **Needs pfs_utils 7.4.44 or later.**

## [1.1.83] - 2026-11-15

- INSTRM-3112 - The spot size, peak and flux sent to mlp1 (`find_representative_spot`) are medians over the detections matched to guide stars, as documented, not over every detection that passed the source filter; a frame with no good match still uses every filtered detection, as before (11 of 20,593 exposures since Run 30). In the Run 30 engineering visits the values change by under 1% in the median (size within 9%, peak and flux within a factor 0.8–1.9 in single exposures of the sparse visits). Called directly, `find_representative_spot` leaves a spot whose moments give no real semi-minor axis out of the size instead of returning NaN when every spot is like that (0 when none is left); the actor's source filter already drops such spots, so this changes nothing on sky. `AgCmd` and the `ag` controller no longer fail at start-up when the actor config has no `tel_status` entry.

## [1.1.82] - 2026-11-15

- INSTRM-3111 - `acquire_field` estimates focus from the detections the autoguide loop would match: its guide stars (galaxies removed, nothing else) and its residual limit (`ag.MAX_RESIDUAL`, 0.2 mm), so acquisition and guiding estimate focus from the same spots. Its offsets, `agc_match` rows and saved files still come from the stricter acquisition catalog and its own limit (0.5 mm unless `max_residual` is given); it matches a second time for focus only (`GuideOffsets.focus_detected_objects`), about 35 ms per call. Replaying the ten Run 30 engineering visits, the acquisition focus then equals the loop's on every camera of all 290 exposures; against 1.1.81 a camera's value changes by up to 5 µm in the dense visits and by a median of 4 µm (up to 117 µm) in the sparse visits 148258–148259. Also, guide stars now load on pandas 3 as well as pandas 2: `tweak_target_position` gives pfs_utils a copy of the parallax column, which pfs_utils clamps in place.

## [1.1.81] - 2026-11-15

- INSTRM-3098 - `acquire_field` estimates focus from the detections matched to guide stars, as the autoguide loop already did, instead of every unflagged detection. With FLAT_TOP retired (INSTRM-2586), faint unmatched spots would otherwise reach the estimate; in the sparse Run 30 visits 148258–148259 they moved a camera's focus by 100–290 µm. A camera without matched spots on both halves of its detector now reports NaN in `focusErrors`, and `guideErrors` carries a NaN focus when no camera has them, as in the autoguide loop; before, unmatched spots could still give a number. The same NaNs are written to `agc_guide_offset` (`guide_delta_z`, `guide_delta_z1`–`guide_delta_z6`), which already holds NaN wherever a camera had no estimate. `ag focus` (no catalog) is unchanged.

- INSTRM-2841 - The `filter_bad_shape` option of `acquire_field` and `autoguide start|initialize|reconfigure` no longer does anything (`ag.FILTER_BAD_SHAPE` and the option plumbing are gone): guiding was the same with the filter on or off, so `BAD_SHAPE` spots are now always excluded from acquisition and guiding (`BAD_SHAPE` joins `BAD_DETECTION_FLAGS`). The flag itself is still set by agcc and used by focus and drp_qa. The keyword is still accepted because the Gen2 AG scripts and iic send it: `filter_bad_shape=yes` gets an inform (it asks for what now always happens) and `filter_bad_shape=no` a warning. It goes once INSTRM-3067 removes it there.
- INSTRM-2586 - `BAD_DETECTION_FLAGS` loses `FLAT_TOP`: the flag is retired (ics_agccActor no longer sets it, and pfs_utils drops it from `SourceDetectionFlags.BAD_DETECTION` while keeping the bit so old `agc_data` rows decode). No change to guiding: every spot it marked was either `SATURATED` or unmatched.
- INSTRM-2931 - `max_correction` is compared with the offset on the sky (`dRA cos(Dec)`, `dDec`), not with the raw RA coordinate offset: at Dec 66 the 10 arcsec limit was 4 arcsec on the sky. The INVALID_OFFSET message and alert now also give the on-sky RA offset.

## [1.1.80] - 2026-10-26

- INSTRM-3057 - Write `camera_enabled` to the new `agc_match.camera_enabled` column (False for a match from a disabled camera) instead of only logging it. Needs spt_operational_database migration `f3a8d61c2b57`.

## [1.1.79] - 2026-10-22

- INSTRM-3058 - `identified_objects.npy` (the ag `data` keyword, exported to FITS by vgwActor) now carries `match_flags` (the `SourceMatchingFlags` bitmask) in place of `matched`. The table keeps its 12 columns; `match_flags == 1` selects the rows `matched == 1` did.

## [1.1.78] - 2026-10-22

- INSTRM-3038 - Write `SourceMatchingFlags` to `agc_match.flags`: `GOOD_MATCH` (still 1) for a match within tolerance that survived rejection, otherwise `NO_MATCH`, `BAD_RESIDUAL` and/or `UNUSED_MULTI_MATCH` instead of 0. `identified_objects.matched` is now `GOOD_MATCH` alone, so a duplicate claim on a star no longer counts as matched (nor enters the representative-spot pool).
- INSTRM-2881 - Adding filter_bad_shape options to autoguide initialize and autoguide start.

## [1.1.73] - 2026-01-23

- INSTRM-2840 - Better error handling for focus when very few stars.
- Add `notebooks` folder to gitignore.

## [1.1.72] - 2026-01-15

- INSTRM-2847 - Check on AG cadence.

## [1.1.71] - 2026-01-08

- INSTRM-2842 - Don't send alert detail.

## [1.1.70] - 2026-01-08

- INSTRM-2838 - Change `BAD_SHAPE` filter logic default.

## [1.1.69] - 2026-01-08

- INSTRM-2792 - Reverse guide/acquisition filtering logic.

## [1.1.68] - 2026-01-08

- INSTRM-2798 - Filter `BAD_SHAPE`.

## [1.1.67] - 2026-01-08

- INSTRM-2832 - Fix matched flag column.

## [1.1.66] - 2025-12-17

- INSTRM-2812 - Fix project metadata. Remove custom DB singleton items.

## [1.1.65] - 2025-12-12

- INSTRM-2812 - Use `pfs_utils` for database access.

## [1.1.64] - 2025-11-24

- INSTRM-2795 - Pull back dataframe and explicitly cast values for telescope status.

## [1.1.63] - 2025-11-17

- INSTRM-2790 - Add exception checking within the `sendAlert` command itself in case it fails. Otherwise it fails silently and kills the control loop. Fix bad quoting in sending error.

## [1.1.62] - 2025-11-13

- INSTRM-2788 - If no detected objects, return nan focus values.

## [1.1.61] - 2025-11-13

- INSTRM-2787 - Adding fix for `taken_at` as float.

## [1.1.60] - 2025-11-13

- INSTRM-2786 - Hotfixes for EngRun25.

## [1.1.59] - 2025-11-11

- Removing check for `design_id` and verbose logging.

## [1.1.58] - 2025-11-10

- Adding missing logger for `sendAlert`.

## [1.1.57] - 2025-11-10

- Hotfix for keywords `filter_bad_shape`.

## [1.1.56] - 2025-11-10

- Hotfix for keywords `visit0`.

## [1.1.55] - 2025-11-10

- Hotfix for keywords `visit0`.

## [1.1.54] - 2025-11-10

- Hotfix for bad vocab keyword.

## [1.1.53] - 2025-11-10

- Remove circular dependency.

## [1.1.52] - 2025-11-10

- Fixing hotfix.

## [1.1.51] - 2025-11-10

- Fixing up datamodel requirement.

## [1.1.50] - 2025-11-10

- Add `pfs_datamodel` to EUPS table list.

## [1.1.49] - 2025-11-10

- INSTRM-2668 - Remove circular import (hotfix for focus routines).

## [1.1.48] - 2025-11-10

- INSTRM-2668 - Hotfix for focus routines.

## [1.1.47] - 2025-11-10

- INSTRM-2668 - Update focus routines. Use consistent defaults.

## [1.1.46] - 2025-11-10

- INSTRM-2713 - Update `get_detected_objects` to support filtering of multiple flags, not just those `< 2`. Rename bad flags. Adding the `filter_bad_shape` option and default it to true.

## [1.1.45] - 2025-11-10

- INSTRM-2715 - Always apply transformations to the guide stars (which makes the `pfs_config_agc` table a bit redundant).

## [1.1.44] - 2025-11-10

- INSTRM-2737 - Adding basic alerts for the control loop. Add Gen2 error reporting.

## [1.1.43] - 2025-11-06

- INSTRM-2715 - Add the `query_pfs_config_agc` function. Use the `visit_id` to look up guide objects via the `pfs_config_agc` table. If no `pfs_config_agc` entry exists, fall back to the entry from `pfs_design_agc` and adjust the coordinates appropriately. Check for `design_id` and stop loop if not present. Passing `visit0` from iic to commands. Better edge case handling. Similar column names regardless of source.
- Clean up the `send_guide_offsets` and use consistently.

## [1.1.42] - 2025-10-30

- INSTRM-2692 - Fix nominal vs centered database write entries.

## [1.1.41] - 2025-10-28

- INSTRM-2746 - Set up testing framework.

## [1.1.40] - 2025-10-28

- INSTRM-2724 - Remove the OTF and SKY modes.
    - Change `autoguide.autoguide` to `autoguide.get_exposure_offsets`.
    - Move the mlp1 guide offset command into the actorCalls utils file.
    - Remove some of the logic dealing with unused options (e.g. `magnitude`, `fit_dinr`, `fit_dscale`, etc.)
    - `query_pfs_design` only returns the ra, dec, and inst_pa columns, which is all that was being used.
    - Remove the `set_design` and `set_design_agc` and getting the `guide_catalog` object directly in the run loop.
    - Change `acquire_field` `altazimuth` param to default `True`
- INSTRM-2665 - Set python minimum to 3.12. Update metadata in pyproject file. Add basic GHA test for pip install.

## [1.1.39] - 2025-09-14

- INSTRM-2686 - Handle timeouts differently from errors. Changing the `TimeoutError` generic check to a specific `RuntimeError` check around the `updateTelStatus` in the main part of the ag loop.

## [1.1.38] - 2025-09-13

- Change `updateTelStatus` timeout to 10 seconds.

## [1.1.37] - 2025-09-13

- Hotfix to change the `MAX_CORRECTION` default back to `10.0` arcsec.

## [1.1.36] - 2025-09-13

- EngRun-Sept-2025-09-12 - EngRun updates from 09-12-2025:
    - Make the NON_BINARY check part of the regular checks for guiding.
    - Don't return a series for a single agc match.
    - Change the `MAX_CORRECTION` default to `0.5` arcsec to match what's on the SA reconfigure screen.
    - Fix the matched guide objects index. See INSTRM-2683

## [1.1.35] - 2025-09-12

- EngRun-Sept-2025 - EngRun updates from 09-11-2025:
    - Fix the "spot size" (seeing) sent to mlp1
    - Send `np.nan` instead of `None` values in `guideErrors`.
    - Update the `MAX_CORRECTION` default to `10.0` arcsec. Change text that is sent.
    - INSTRM-2680 - Don't break from the run loop during error, just set mode to `STOP`.
    - Change how `identified_objects` table is built so it uses a DataFrame.
    - Change the logic of `is_acuisition` to `is_guide` and set it so we filter more during guiding.

## [1.1.34] - 2025-09-11

- INSTRM-1057 - Replace IERS handling with tools from `ics_utils`.
- INSTRM-2469 - Repository cleanup:
    - Reorganize code structure
    - Rename functions and directories (including `kawanomoto` to `coordinates`)
    - Standardize imports
    - Remove unused code
    - Format with black
    - Use pfs_utils for AutoGuiderStarMask
- INSTRM-2548 - Apply appropriate filters during acquisition and guiding. Fix "seeing" calculation.
- INSTRM-2567 - Implement `MAX_CORRECTION` to control if offsets are used or not.
- INSTRM-2573 - Stop run loop whenever an exception is raised.
- INSTRM-2598 - Improve agc_match insertion.
- INSTRM-2613 - Don't hard-code gaia dsn string.
- INSTRM-2615 - Use ics-utils DB for opdb.
- INSTRM-2625 - Fix a logical error in conditional check for spots
- INSTRM-2630 - Large refactor of AG code. See commit message for details.
- INSTRM-2642 - Clean up build tools.
- INSTRM-2647 - Make `guide_stars` format consistent across lookup methods.
- INSTRM-2650 - Clean up database interaction.
- INSTRM-2656 - Set minimum python version to `3.11`.
- INSTRM-2667 - Update the output of shared numpy files for vgw.
- INSTRM-2669 - Add `status` column to `guideErrors` to indicate if offset correction is valid.
- Adding generic query_db function which can return a dataframe

## [1.1.33] - 2025-07-09

- INSTRM-2607 - Moving `coordinates.py` to `pfs_utils`.

## [1.1.32] - 2025-06-20

- Hotfix for INSTRM-2602 - fix import path

## [1.1.31] - 2025-06-20

- INSTRM-2603 - Clean up merge conflicts from 10-2024

## [1.1.30] - 2025-06-19

- INSTRM-2471 - Consolidate usage of Subaru_POPT2_PFS

## [1.1.29] - 2025-06-19

- INSTRM-2602 - reverts INSTRM-2599 (`1.1.281`)

## [1.1.28] - 2025-06-19

- INSTRM-2599 - Only use good guide stars for now

## [1.1.27] - 2025-06-18

- INSTRM-2594 - Return all AG filtering flags, not just the first. Avoid repeat calls to filtering.

## [1.1.26] - 2025-05-26

- Fixup

## [1.1.25] - 2025-05-26

- INSTRM-2563: updateTelStatus stray fixes.

## [1.1.24] - 2025-05-24

- INSTRM-2555: send our visit to gen2 updateTelStatus

## [1.1.23] - 2025-05-24

- INSTRM-2553: raise agcc exposure timeout

## [1.1.22] - 2025-05-23

- INSTRM-2551 - Fixing compatibility with old design files.

## [1.1.21] - 2025-05-23

- INSTRM-2552 - Revert the systematic RA offset introduced in INSTRM-2522

## [1.1.20] - 2025-05-22

- tickets/INSTRM-2464 - adding extra column for guide_star_flag

## [1.1.9] - 2025-05-21

- remove setup.py

## [1.1.8] - 2025-05-21

- Re-tag of 1.1.7 - tickets/INSTRM-2547

## [1.1.7] - 2025-05-21

- Updates from Kawanamoto-san - kawanomoto_20250522_1

## [1.1.6] - 2025-05-21

- tickets/INSTRM-2543 - removing print statements and add much logging

## [1.1.5] - 2025-05-21

- tickets/INSTRM-2541 - adding checks for flags when detecting no results.

## [1.1.4] - 2025-05-21

- Merged tickets/INSTRM-2449b, which adds support for HSC guide stars and filtering.

## [1.1.3] - 2025-05-20

- INSTRM-2449b - branch tip

## [1.1.2] - 2025-05-19

- INSTRM-2522: correct ag pointing error

## [1.1.1] - 2025-03-23

- Applying updates from kawanomoto_20250323_1 with small changes to `agarray2momentdifference`

## [1.1.0] - 2025-03-14

- INSTRM-2477: make ag-ics and ics_agActor normal ICS

## [1.0.1] - 2025-03-04

- making a normal tag for tags/kawanomoto_2025-01-23_04

## [EngRun20] - 2025-07-24

- Merge pull request #2 from Subaru-PFS/tickets/INFRA-339

## [pre-EngRun20] - 2025-07-24

- Use user 'gen2' for 'gaia3' table

## [subaru-telescope-202205] - 2022-05

- Merge pull request #2 from Subaru-PFS/tickets/INFRA-339

## [subaru-telescope-202108] - 2021-08

- Emit object metadata as numpy .npy format files
