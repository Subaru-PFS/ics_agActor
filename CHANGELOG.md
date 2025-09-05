# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- INSTRM-1057 - Replace IERS handling with tools from `ics_utils`.
- INSTRM-2469 - Repository cleanup:
    - Reorganize code structure
    - Rename functions and directories (including `kawanomoto` to `coordinates`)
    - Standardize imports
    - Remove unused code
    - Format with black
    - Use pfs_utils for AutoGuiderStarMask
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
