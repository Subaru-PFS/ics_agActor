# AGENTS.md — ics_agActor

Guidance for AI coding agents (GitHub Copilot, Claude, GPT-based tools, etc.) working in this
repository. Read this file **before** writing or modifying any code.

---

## Project overview

`ics_agActor` is a Python actor that controls the **Auto Guider (AG)** subsystem of the
[Subaru Prime Focus Spectrograph (PFS)](https://pfs.ipmu.jp/). It runs inside the Subaru
Instrument Control System (ICS) and is responsible for:

- **Field acquisition** — coarsely matching detected AG sources to the guide-star catalog and
  deriving an initial telescope offset.
- **Auto-guiding** — maintaining pointing accuracy during an exposure by computing per-frame
  RA/Dec/InR/scale corrections.
- **Focus monitoring** — converting second-moment differences (spider-shadow proxy) into focus
  error estimates in mm.

The ICS framework is built on [`tron_actorcore`](https://github.com/Subaru-PFS/tron_actorcore) /
`sdss-actorcore`. The actor communicates over the Tron message bus with the AG Camera Controller
(`ics_agccActor`) and the Subaru Gen2 observation-control system.

---

## Language and runtime

| Item | Value |
|------|-------|
| Language | Python 3.12 |
| Package manager | `uv` (lockfile: `uv.lock`); `pip` also works |
| Build system | `setuptools` + `lsst-versions` (`pyproject.toml`) |
| Test runner | `pytest` |
| Linter | `ruff` |
| Docstring convention | NumPy style |

---

## Repository layout

```
python/agActor/
  Commands/
    AgCmd.py               # Tron command vocabulary + command handlers (~950 lines, god-object)
  Controllers/
    ag.py                  # AG state machine, exposure loop, alert dispatch
  catalog/
    astrometry.py          # Astrometry integration (DEAD CODE — not imported in main path)
    gen2_gaia.py           # Gaia catalog interface (potentially dead)
    pfs_design.py          # PFS design/config file handling
  coordinates/
    FieldAcquisitionAndFocusing.py  # High-level field-acquisition and focus orchestration
    Subaru_POPT2_PFS_AG.py          # Core astrometric engine (PFS class)
  models/
    agcc.py / gen2.py / mlp1.py     # Tron model wrappers
  utils/
    actorCalls.py          # Tron actor communication helpers
    data.py                # Dataclasses (GuideCatalog, GuideOffsets); re-exports GuideOffsetFlag, BAD_DETECTION_FLAGS
    queries.py             # All query_* / write_* DB functions, get_telescope_status, get_detected_objects, search_gaia; defines GuideOffsetFlag, BAD_DETECTION_FLAGS
    guide_catalog.py       # get_guide_objects, filter_guide_objects, tweak_target_position
    focus.py               # Focus utilities
    logging.py             # Logging helpers
    math.py                # Mathematical utilities
    plot.py                # Plotting helpers
    telescope_center.py    # Telescope centering utilities
    to_altaz.py            # Altitude/azimuth coordinate conversion utilities
  autoguide.py             # get_exposure_offsets — per-frame guiding entry point
  field_acquisition.py     # acquire_field — field-acquisition entry point
  main.py                  # Actor entry point
tests/                     # pytest suite (mostly Jupyter notebooks via nbval)
data/                      # Sample FITS files for testing
```

---

## Branching and commit conventions

- **Default development branch:** `main` (GitHub default). `master` is a legacy alias kept for
  EUPS compatibility.
- **Feature branches:** `tickets/INSTRM-NNNN` for JIRA tickets; `refactor/<short-name>` for
  larger refactors (e.g. `refactor/pfs-astrometric-engine`); `u/<author>/<topic>` for personal
  work-in-progress branches.
- **Commit messages:** follow conventional-commits style where possible
  (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`). Include the JIRA ticket number in
  the message body or as a prefix when one exists (e.g. `INSTRM-2881`).
- **Pull requests:** open against `main`. Squash-merge is preferred for ticket branches; merge
  commits are used for larger refactor branches to preserve history.

---

## Development setup

```bash
# Clone and enter the repo
git clone https://github.com/Subaru-PFS/ics_agActor.git
cd ics_agActor

# Create a virtual environment and install with dev extras
uv venv
uv pip install -e ".[dev]"

# Or with plain pip
pip install -e ".[dev]"
```

> **Note:** Several dependencies (`pfs-utils`, `pfs-datamodel`, `ics-utils`, `tron_actorcore`)
> are installed directly from GitHub. A working internet connection (or pre-cached wheels) is
> required for the initial install.

---

## Running tests

```bash
pytest                          # all tests (includes nbval notebook tests)
pytest tests/test_version.py    # a single file
pytest --no-header -q           # quiet output
```

The test suite uses `--nbval` to execute Jupyter notebooks in `tests/`. Notebooks hit the
operational database; set environment variables (or use a local SQLite fixture) as needed.

---

## Linting

```bash
ruff check python/               # lint
ruff check --fix python/         # auto-fix safe issues
```

Ruff is configured in `pyproject.toml`. Naming rules N802/N803/N806/N815/N816 are **intentionally
suppressed** because the codebase uses camelCase function and variable names that match
the upstream Subaru/PFS convention. Do not re-enable those rules.

---

## Key files an agent should understand first

| File | Why |
|------|-----|
| `python/agActor/coordinates/Subaru_POPT2_PFS_AG.py` | Core astrometric engine. `PFS.RADECInRShiftA` is the central routine: coarse Cramer's rule match → refined NN match → least-squares solve → iterative outlier rejection. Read the inline Phase comments and the module-level docstring before touching it. |
| `python/agActor/coordinates/FieldAcquisitionAndFocusing.py` | Orchestrates the `PFS` class; calls `makeBasis`, `RADECInRShiftA`, `agarray2momentdifference`. |
| `python/agActor/utils/data.py` | Dataclasses (`GuideCatalog`, `GuideOffsets`). Re-exports `GuideOffsetFlag` and `BAD_DETECTION_FLAGS` from `queries.py` for backward compatibility. ~370 lines. |
| `python/agActor/utils/queries.py` | All `query_*` / `write_*` DB functions, `get_telescope_status`, `get_detected_objects`, `search_gaia`. Canonical home of `GuideOffsetFlag` and `BAD_DETECTION_FLAGS` (defined here; `data.py` re-exports them). |
| `python/agActor/utils/guide_catalog.py` | `get_guide_objects`, `filter_guide_objects`, `tweak_target_position`. |
| `python/agActor/Commands/AgCmd.py` | ~950-line god-object. Tron command vocabulary + all command-handler logic. |
| `python/agActor/Controllers/ag.py` | AG state machine and exposure loop (~600 lines). |
| `REFACTORING.md` | Prioritised list of known code-quality issues with concrete fix suggestions. Check this before making structural changes. |
| `plans/plan-refactorRADECInRShiftA.prompt.md` | Step-by-step refactoring plan specifically for `Subaru_POPT2_PFS_AG.py`. Refer to the **Progress** table at the top to see which steps are already done. |

---

## Coding conventions

1. **NumPy docstrings** on all public functions and methods. Parameters, returns, and notes
   sections are required; examples are optional but welcome.
2. **Type annotations** on all new function signatures. Use `np.ndarray` for arrays; add shape
   comments `# (N, 8)` where the shape is non-obvious. Use python 3.12+ style.
3. **Line length:** 110 characters (Black + isort profiles are configured).
4. **No magic numbers.** Named constants belong in a `constants.py` module or, for
   algorithm-specific values, in a clearly commented module-level variable.
5. **No silent exception swallowing.** DB write failures must be logged *and* re-raised (or
   surfaced to the caller via a `raise_on_error` param). See REFACTORING.md Issue 12.
6. **`or` on numeric fields is a bug.** Use `if x is None: x = default` instead of `x = x or
   default`. See REFACTORING.md Issue 7.
7. **Do not use the Python 2 unbound-method idiom** (`ClassName.method(self, ...)`). Use
   `self.method(...)` instead.
8. **Camel-case names** (`RADECInRShiftA`, `makeBasisPfi`, `agarray`) are preserved for
   compatibility with upstream PFS library conventions. New helpers introduced by agents should
   use `snake_case`.
9. **Dataclasses over magic-index arrays.** When a function returns multiple named values, prefer
   a `@dataclass` or `NamedTuple` over a raw `np.block(...)` matrix accessed by positional index.

---

## Active refactoring work

### `Subaru_POPT2_PFS_AG.py` — `PFS` astrometric engine

See `plan-refactorRADECInRShiftA.prompt.md` for the full step-by-step plan.

| Step | Status |
|------|--------|
| Step 1 — Add docstrings and inline comments | ✅ Done (`686f712`) |
| Step 2 — Rename opaque variables | ✅ Done (`9605a6c`) |
| Step 3 — `N_AG_CAMERAS` constant, `np.full` | ⏳ Pending |
| Step 4 — Remove dead `v_a`, inline `makeBasisPfi` | ⏳ Pending |
| Step 5 — Extract `_unpack_catalog` helper | ⏳ Pending |
| Step 6 — Extract `_select_by_detector_half` helper | ⏳ Pending |
| Step 7 — Extract `_build_basis_columns` / `_extract_offsets` | ⏳ Pending |
| Step 8 — Extract `_iterative_outlier_rejection`, add `MatchResult` | ⏳ Pending |

When working on this file: **do not start a pending step until the previous step's changes are
committed and tests pass** (see Further Consideration 3 in the plan — regression tests should
be added before Steps 3+).

### General codebase (from REFACTORING.md)

High-priority items not yet addressed:

| Issue | File(s) | Status |
|-------|---------|--------|
| Issue 1 — `apply_coord_deltas` helper | `field_acquisition.py`, `autoguide.py`, `utils/data.py` | ⏳ |
| Issue 2 — `build_filter_flags` helper | `AgCmd.py`, `field_acquisition.py`, `autoguide.py` | ⏳ |
| Issue 3 — Replace `**kwargs` tunnel with typed params | `AgCmd.py` → `ag.py` → `field_acquisition.py` / `autoguide.py` | ⏳ |
| Issue 4 — Slim down `AgCmd.py` | `Commands/AgCmd.py` | ⏳ |
| Issue 5 — Extract `AgExposureLoop` | `Controllers/ag.py` | ⏳ |
| ~~Issue 6 — Split `utils/data.py`~~ | ~~`utils/data.py`~~ | ✅ `refactor/utils-cleanup` |
| Issue 8 — `BaseModel` for model classes | `models/*.py` | ⏳ |
| Issue 11 — `constants.py` | scattered | ⏳ |

---

## Known gotchas and traps

| # | Location | Description |
|---|----------|-------------|
| G1 | `FieldAcquisitionAndFocusing.py` | `match_result` columns (0–9) are accessed by magic index. Column 6 = resid_x, 7 = resid_y, 8 = is_inlier, 9 = cat_index. A `MatchResult` dataclass is planned in Step 8. |
| G2 | `Subaru_POPT2_PFS_AG.py` | `right_detector_mask` is computed twice inside `RADECInRShiftA`; the second computation is redundant. |
| G3 | `Subaru_POPT2_PFS_AG.py` | `rejection_threshold == rejection_threshold_old` uses exact float equality; safe today but fragile if the threshold ever becomes `nan`. Fix planned in Step 8. |
| G4 | `utils/data.py` | ~~`taken_at or db_taken_at` (and similar for `inr`, `adc`, `m2_pos3`) silently replaces a valid caller-supplied `0.0`.~~ **Fixed** (`refactor/fix-or-on-numeric-fields`): all `or` shorthand replaced with explicit `is None` guards. |
| G5 | `Controllers/ag.py` | `Params` inner class (lines 44–80) is the seed of the future typed-params object. Do not duplicate it. |
| G6 | `catalog/astrometry.py` | **Dead code.** Do not add imports or callers for this file. |
| G7 | `utils/data.py` / `utils/actorCalls.py` | ~~DB write exceptions are caught and logged but not re-raised in older paths.~~ **Fixed** (`refactor/reraise-swallowed-exceptions`): `write_agc_guide_offset`, `write_agc_match`, and `sendAlert` all re-raise after logging. |
| G8 | `utils/data.py` | `GuideOffsets.save_numpy_files()` defaults `base_dir="/dev/shm"`, which silently fails on macOS. |
| G9 | `ag_common.py` / logs | `guideErrors` has three status values: `OK`, `ERROR`, `INVALID_OFFSET`. Before the fix in this session, `_GUIDE_ERR` only matched `OK\|ERROR`, silently dropping all `INVALID_OFFSET` frames from the dashboard. Always use `(\w+)` as the status group. |
| G10 | logs / `gen2.py` | `rot` in `tel_axes` lines is the **parallactic/field angle** (changes at ~0.04°/min with sky tracking, range ~20–55°). `inr` in `data.py:497` lines is the **instrument rotator mechanical position** (±180°, jumps between visits). These differ by tens of degrees and must not be treated as the same quantity. |
| G11 | `data.py:497` log line | `taken_at=…,inr=…,adc=…,m2_pos3=…` is only emitted during `acquire_field`, not during regular `autoguide` frames. Per-frame `inr` for science exposures comes from the DB (`agc_exposure_info.insrot`), not the log. |

---

## Testing guidance

- **Regression first:** Before refactoring any numerical routine (especially in
  `Subaru_POPT2_PFS_AG.py` or `FieldAcquisitionAndFocusing.py`), add a pytest test that captures
  the output on known input data (FITS files in `data/`). This guards against silent numerical
  drift.
- **Notebook tests:** The `tests/*.ipynb` files run via `nbval`. Ensure notebooks can execute
  end-to-end with a clean DB fixture before modifying them.
- **DB-dependent tests:** Use the `tests.db` SQLite fixture (`populate_test_db.py`) for local
  development; do not require a live opDB connection.

---

## AG operation concepts

This section captures domain knowledge about how the AG subsystem operates, derived from
analysis of operational logs and the codebase. It is essential context for any work on
`ag_common.py`, the dashboard notebook, or the command handlers.

### Observation sequence and guiding modes

Every observation block follows this lifecycle, driven by commands from Gen2/OCS:

```
acquire_field  →  autoguide start  →  autoguide reconfigure visit=N  →  autoguide stop
   (acquire)          (converge)              (autoguide)                   (idle)
```

| Mode | Trigger log line | Description |
|------|-----------------|-------------|
| `acquire` | `new cmd: acquire_field` | Coarse field-acquisition: telescope is far off, offsets can be tens of arcseconds. |
| `converge` | `new cmd: autoguide start` | Convergence visit: `visit0 == visit_id`, meaning this visit *is* the reference config. The AG applies corrections each frame but the telescope often needs several frames to settle. Offsets are typically 1–20″. |
| `autoguide` | `new cmd: autoguide reconfigure visit=N` | Science visit N begins. Steady-state guiding mode. Residuals should be sub-arcsecond. |
| `idle` | `new cmd: autoguide stop` | AG is stopped between blocks. |
| `unknown` | (before any command) | Parser default before the first command is seen. |

**Key distinction — converge vs acquire:** Both have large offsets, but they are different phases.
`acquire` is a single coarse-match frame. `converge` is a multi-frame settling sequence before
science begins. Filter dashboards and diagnostics on `mode == 'autoguide'` to see only steady-state
science guiding. The OCS observation log (`obslog`) records convergence visits explicitly.

**`autoguide reconfigure` without `visit=`** is a *parameter-only* change (e.g. `max_correction`,
`filter_bad_shape`). It does **not** change mode or visit_id. It emits a `ReconfigureRecord` in the
parser and appears as a dotted vertical line in the dashboard.

### Per-frame record identity

Every data record (guide solution, focus, star stats, matched count, camera counts, telescope state)
carries both:
- `visit_id` — PFS visit ID active when the record was emitted. Tracked from any `new cmd:` line
  carrying `visit_id=N`; `autoguide reconfigure` uses `visit=N` instead.
- `frame_id` — AGC exposure identifier (e.g. `1047858`). Set from the `agc_frameid="N"` reply line
  that arrives from `agccActor` *before* the data records for that frame.

To locate a specific moment: "During visit V, frame F, what was the guider state?" — filter
records by both fields.

### guideErrors line — the per-frame guide record

Format:
```
guideErrors=<frame_id>,<dRA>,<dDec>,<dInR>,<dAz>,<dEl>,<dFocus>,<dScale>,<status>
```

All offsets are in arcseconds except `dFocus` (mm) and `dScale` (dimensionless).

`status` has three values:

| Status | Meaning |
|--------|---------|
| `OK` | Correction computed and sent to the telescope. |
| `ERROR` | Computation failed (e.g. too few matched stars). |
| `INVALID_OFFSET` | Correction computed but **not applied** because at least one axis exceeded `max_correction`. The offset values in the record are the computed (unclamped) values — useful for diagnostics. |

`INVALID_OFFSET` frames are common at the start of convergence sequences and can persist for many
frames if `max_correction` is set too conservatively for the actual pointing error. A run of many
consecutive `INVALID_OFFSET` frames in `autoguide` mode indicates a serious pointing problem (the
AG is computing corrections it cannot apply, and the telescope drifts uncorrected).

### max_correction

Set via `autoguide reconfigure max_correction=N` (arcseconds, absolute, per-axis). The default
operational value is `10` arcsec. Any frame where the computed correction on any axis exceeds this
value gets status `INVALID_OFFSET` and no correction is sent.

In the dashboard, `max_correction` is shown as dotted ±N arcsec horizontal reference lines on the
Guide Offsets and InR panels, drawn as a step function that updates when reconfigure events occur.

### Telescope position quantities

Three distinct angle quantities appear in the logs — they are **not interchangeable**:

| Quantity | Source | Log pattern | Cadence | Description |
|----------|--------|-------------|---------|-------------|
| `az`, `el` | Gen2 `tel_axes` | `reply=ag.ag … gen2 I tel_axes=<az>,<el>,<rot>,<airmass>` | ~10 s | Telescope azimuth and elevation (degrees). |
| `rot` | Gen2 `tel_axes` (3rd field) | same line | ~10 s | **Parallactic / field rotation angle.** Changes smoothly at ~0.04°/min as the field tracks across the sky. Range roughly 20–55° during a typical night. **Not** the instrument rotator mechanical position. |
| `inr` | `data.py:497` | `taken_at=…,inr=<angle>,adc=<angle>,m2_pos3=<mm>` | per frame | **Instrument rotator real angle** (mechanical position, degrees). Jumps between visits as the rotator slews to the required PA. Range typically ±180°. |
| `adc` | `data.py:497` | same line | per frame | ADC prism position angle (degrees). |
| `m2_pos3` | `data.py:497` | same line | per frame | M2 secondary mirror Z-position (mm). Typical value ~4.77 mm; changes only with M2 focus corrections. |

**`rot` vs `inr`:** The difference between these two quantities is not constant — it varies by tens
of degrees across a night and between visits. `rot` represents the sky's parallactic evolution;
`inr` is where the rotator physically sits. Do not treat them as equivalent.

**`data.py:497` cadence caveat:** The `taken_at=…,inr=…` line is currently emitted only during
`acquire_field` calls (~36 records/night in typical use), not during regular `autoguide` frames.
Per-frame `inr` for autoguide exposures must be retrieved from the database
(`agc_exposure_info.insrot`).

### AG dashboard notebook (`notebooks/ag_explorer.ipynb`)

Shared parser: `notebooks/ag_common.py`. Record types emitted by `parse_line`:

| Record type | Dataclass | Key fields |
|-------------|-----------|------------|
| `guide` | `GuideRecord` | `frame_id`, `visit_id`, `mode`, `status`, `dra/ddec/dinr/daz/del_/dfocus/dscale` |
| `focus` | `FocusRecord` | `z1`–`z6` per-camera focus offsets (mm), `visit_id`, `frame_id`, `mode` |
| `star_stats` | `StarStatsRecord` | `flux`, `peak`, `size` (seeing proxy), `visit_id`, `frame_id`, `mode` |
| `matched` | `MatchedRecord` | `count` matched sources, `visit_id`, `frame_id`, `mode` |
| `camera_counts` | `CameraCountRecord` | per-camera object counts list, `visit_id`, `frame_id`, `mode` |
| `tel_axes` | `TelAxesRecord` | `az`, `el`, `rot` (parallactic angle from Gen2, ~10 s cadence) |
| `tel_rot` | `TelRotRecord` | `pa`, `insrot` (sky PA and mechanical rotator angle from Gen2 `tel_rot` reply lines, ~1 s cadence — use this for high-cadence InR tracking) |
| `tel_state` | `TelStateRecord` | `inr`, `adc`, `m2_pos3` (acquire-only — see caveat above) |
| `visit_changes` | `VisitChangeRecord` | `visit_id`, `t` |
| `design_changes` | `DesignChangeRecord` | `design_id`, `t` |
| `reconfigs` | `ReconfigureRecord` | `params` dict, `visit_id`, `mode` (param-only reconfigure events) |

All per-exposure records carry `visit_id`, `frame_id`, and `mode` so any record can be
cross-referenced with any other by frame or visit.

---

## Interactive diagnostics

### AG Explorer dashboard

`notebooks/ag_explorer.ipynb` is a Panel/Plotly dashboard for interactive exploration of AG
log files. Serve it with:

```bash
panel serve notebooks/ag_explorer.ipynb --autoreload
```

Select a log file from `notebooks/logs/`, choose panels and time windows, and use the checkboxes
to toggle file size filtering and non-science frame display. See the "AG operation concepts"
section above for the mode taxonomy and record types used by the dashboard parser.

### query-ag-data CLI

`query-ag-data` (installed console script) is the go-to tool for querying AG data from the operational
database during interactive diagnostic work (e.g. from a Jupyter notebook or the terminal).
The implementation lives in `python/agActor/utils/diagnostics.py`.

- **CLI usage:** `query-ag-data <design_id> [--frames …] [--table …] [--format json|csv] [--save]`
- **Notebook / script usage:** import `query_ag_data` directly and get back a `dict[str, pd.DataFrame]`:

  ```python
  from agActor.utils.diagnostics import query_ag_data
  tables = query_ag_data(design_id=0x00112233aabbccdd)
  tables["offsets"]   # ra_offset, dec_offset, … per frame
  tables["detected"]  # raw spot detections
  ```

- Tables available: `exposure_info`, `detected`, `matched`, `offsets`.
- Connects to `pfsa-db` as `public_user` by default; pass an open `OpDB` instance via the
  `opdb` parameter to reuse an existing connection.

---

## Out-of-scope / do not touch

- `ups/ics_agActor.table` — EUPS table file. Edit only if package dependencies change.
- `SConstruct` — SCons build file for EUPS. Do not modify unless explicitly asked.
- `catalog/astrometry.py` and `catalog/gen2_gaia.py` — potentially dead; do not add new callers.
  Deletion should be a deliberate, reviewed change.
