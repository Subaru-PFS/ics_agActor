# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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
