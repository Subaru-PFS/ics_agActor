# ics_agActor - ICS Actor for the Auto Guider

## Overview

The `ics_agActor` is a critical component of the Instrument Control System (ICS) for the Subaru Prime Focus Spectrograph (PFS). It serves as the control actor for the Auto Guider (AG) system, which is responsible for maintaining accurate telescope pointing during observations.

This actor interfaces with the [AG Camera Controller (ics_agccActor)](https://github.com/Subaru-PFS/ics_agccActor), the Subaru Gen2 observation control system, and other components to provide automated guiding functionality for the PFS instrument.

## Features

- **Field Acquisition**: Identifies and acquires guide stars in the field of view
- **Auto-Guiding**: Maintains accurate telescope pointing during observations
- **Guide Star Selection**: Integrates with the GAIA star catalog for guide star selection. Note that most guide stars come directly from the PfsConfig files.
- **Astrometry**: Performs astrometric calculations for accurate pointing. Note that this module is not frequently used and may be removed in the future.
- **Telescope Control**: Calculates and applies pointing corrections
- **Database Integration**: Stores and retrieves observation data from the operational database

## Installation

### Requirements

- Python 3.12 or higher
- Dependencies listed in `pyproject.toml`

### EUPS Installation with LSST Stack

This package uses the Extended Unix Product System (EUPS) for dependency management and environment setup, which is part of the LSST Science Pipelines software stack. The LSST stack is a comprehensive framework for astronomical data processing that provides powerful tools for image processing, astrometry, and data management.

1. Ensure you have the LSST stack installed on your system. If not, follow the installation instructions at the [LSST Science Pipelines documentation](https://pipelines.lsst.io/install/index.html).

2. Once the LSST stack is set up, declare and setup this package using EUPS:
   ```
   eups declare -r /path/to/ics_agActor ics_agActor git
   setup -r /path/to/ics_agActor
   ```

3. The package's EUPS table file (`ups/ics_agActor.table`) will automatically set up the required dependencies within the LSST stack environment:
   - ics_actorkeys
   - ics_utils
   - pfs_instdata
   - pfs_utils

### Standard Setup

Alternatively, you can install the package using pip:

1. Clone the repository:
   ```
   git clone https://github.com/Subaru-PFS/ics_agActor.git
   ```

2. Install the package:
   ```
   cd ics_agActor
   pip install -e .
   ```

## Usage

The actor is typically started as a service in the PFS ICS environment using the `sdss-actorcore` tools.

## Exposure Pipeline

The AG actor has two operational modes that share a common exposure pipeline (`exposure.py`):

### Field Acquisition (one-shot)

Performed via the `acquire_field` command (`AgCmd.acquire_field`). This is a one-off procedure used to initially find the field. It:

- Fetches its own guide catalog internally (with `is_guide=False`) for this single frame.
- Does **not** apply correction range checking (`max_correction=None`) because the initial offset from the expected field center can be large.
- Whether corrections are actually sent to the telescope is controlled by the user's `guide` keyword (`send_offsets` parameter).

### Autoguiding (continuous loop)

Performed via `AgThread.run` in the AG controller. This is the continuous guiding mode that takes consecutive exposures and computes incremental offsets. It:

- Pre-loads a `GuideCatalog` once (with `is_guide=True`) before the guide loop starts, and reuses it across all iterations.
- Each iteration only re-fetches telescope status and detected objects for the current frame.
- Applies correction range checking (`max_correction`, default 10″) — corrections that exceed this threshold are rejected and an alert is raised.
- Always sends corrections to the telescope when the offset is within the allowed range.

### Shared Pipeline Steps

Both modes delegate to `run_exposure_pipeline()` in `exposure.py`, which executes the following steps in order:

1. **Take AGCC exposure** — build and queue the AG camera command.
2. **Gather telescope state** — at mid-exposure, collect status from mlp1/gen2/opDB as configured.
3. **Wait for exposure** — block until the AGCC exposure completes.
4. **Compute `taken_at` timestamp** — calculate the effective observation time.
5. **Compute guide offsets** — either via `field_acquisition.acquire_field` (acquisition) or `autoguide.get_exposure_offsets` (guiding).
6. **Report results** — emit `cmd.inform` messages with offset values and save NumPy files.
7. **Range check & send corrections** — validate offsets against `max_correction` (if set) and send to the telescope.
8. **Compute focus** — calculate focus offset and tilt from matched detected objects.
9. **Write to opDB** — persist guide offsets and match results to the operational database.

## Project Structure

- `python/agActor/`: Main source code
  - `main.py`: Actor entry point and core functionality
  - `exposure.py`: Shared AG exposure pipeline (see [Exposure Pipeline](#exposure-pipeline))
  - `config.py`: Shared configuration parsed from `actorConfig`
  - `autoguide.py`: Auto-guiding offset computation (uses pre-loaded guide catalog)
  - `field_acquisition.py`: Field acquisition offset computation (fetches its own guide catalog)
  - `Commands/`: Command handlers for actor commands
  - `Controllers/`: Hardware controllers
  - `catalog/`: Star catalog integration
    - `astrometry.py`: Astrometric calculations
    - `gen2_gaia.py`: GAIA catalog interface
    - `pfs_design.py`: PFS design file handling
  - `coordinates/`: Coordinate transformations
    - `FieldAcquisitionAndFocusing.py`: Field acquisition and focusing utilities
    - `Subaru_POPT2_PFS_AG.py`: Subaru telescope coordinate transformations
  - `models/`: System component models
    - `agcc.py`: AG Camera Controller model
    - `gen2.py`: Gen2 system model
    - `mlp1.py`: MLP1 model
  - `utils/`: Utility functions
    - `actorCalls.py`: Actor communication utilities
    - `data.py`: Data handling utilities
    - `focus.py`: Focus-related utilities
    - `logging.py`: Logging utilities
    - `math.py`: Mathematical utilities
    - `opdb.py`: Operational database interface
    - `telescope_center.py`: Telescope centering utilities

## Dependencies

- **astropy**: Astronomical calculations
- **fitsio**: FITS file handling
- **numpy**: Numerical computations
- **psycopg2-binary**: PostgreSQL database connector
- **sdss-opscore**: SDSS operations core framework
- **sdss-actorcore**: SDSS actor framework
- **pfs-utils**: PFS utilities

## Development

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Versioning

This project follows semantic versioning. See the [CHANGELOG.md](CHANGELOG.md) for version history and recent changes.

## License

This project is part of the Subaru Prime Focus Spectrograph (PFS) project and is subject to the licensing terms of the PFS collaboration.

## Contact

For questions or issues related to this software, please contact the PFS software team or create an issue in the repository.
