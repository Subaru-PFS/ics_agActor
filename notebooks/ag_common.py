"""
ag_common.py — Shared helpers for AG log parsing and data buffering.

Extracted from the former Matplotlib dashboard module to be used by
`ag_explorer.ipynb` (Plotly/Panel) and any other tooling needing the
same parsing and rolling DataStore utilities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import re
import threading


# Timezone for Hawaii Standard Time (no DST)
HST = timezone(timedelta(hours=-10))


# ───────────────────────────── Data Record Types ──────────────────────────────

@dataclass
class GuideRecord:
    """Per-exposure guide solution parsed from `guideErrors=` lines.

    Attributes
    ----------
    t : datetime
        Timestamp (UTC) parsed from the log line prefix.
    frame_id : int
        AGC exposure (frame) identifier.
    dra, ddec, dinr, daz, del_ : float
        Guide corrections in arcseconds.
    dfocus : float
        Focus offset (mm).
    dscale : float
        Scale correction (dimensionless).
    visit_id : int | None
        PFS visit ID active when this record was emitted.
    mode : str
        Guiding mode: ``'acquire'``, ``'autoguide'``, ``'idle'``, or ``'unknown'``.
    """

    t: datetime
    frame_id: int
    dra: float
    ddec: float
    dinr: float
    daz: float
    del_: float
    dfocus: float
    dscale: float
    status: str = "OK"
    visit_id: int | None = None
    mode: str = "unknown"


@dataclass
class FocusRecord:
    """Per-camera focus offsets (mm) from `agc_guide_offsets` write lines."""

    t: datetime
    z: float
    z1: float
    z2: float
    z3: float
    z4: float
    z5: float
    z6: float
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class StarStatsRecord:
    """Calculated star statistics (flux, peak, PSF size)."""

    t: datetime
    flux: float
    peak: float
    size: float
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class MatchedRecord:
    """Count of matched sources with valid residuals for an exposure."""

    t: datetime
    n: int
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class CameraCountRecord:
    """Detected sources per camera (1–6) for an exposure.

    `counts[i]` is `None` if the camera is absent in that exposure.
    """

    t: datetime
    counts: tuple
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class TelAxesRecord:
    """Telescope Az/El/parallactic-rot sample parsed from Gen2 ``tel_axes`` status lines."""

    t: datetime
    az: float
    el: float
    rot: float  # parallactic / field rotation angle (deg), changes continuously with sky tracking


@dataclass
class TelRotRecord:
    """Physical instrument rotator angle sampled from Gen2 ``tel_rot`` reply lines.

    These are emitted every Gen2 status cycle (roughly once per second during
    active observing), giving a high-cadence view of how the rotator tracks the
    parallactic angle throughout the night.

    Attributes
    ----------
    t : datetime
        Timestamp (UTC) of the log line.
    pa : float
        Sky position angle (degrees) reported by Gen2.
    insrot : float
        Physical instrument rotator angle (degrees).
    """

    t: datetime
    pa: float
    insrot: float


@dataclass
class TelStateRecord:
    """Per-frame telescope state: rotator angle, ADC angle, M2 position.

    Parsed from ``data.py:497`` lines emitted once per AGC exposure.

    Attributes
    ----------
    t : datetime
        Timestamp (UTC) of the log line.
    inr : float
        Instrument rotator real angle (degrees).
    adc : float
        ADC position angle (degrees).
    m2_pos3 : float
        M2 focus position Z3 (mm).
    visit_id, frame_id : int | None
        Active PFS visit and AGC frame at the time of the record.
    mode : str
        Guiding mode active when this record was emitted.
    """

    t: datetime
    inr: float
    adc: float
    m2_pos3: float
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class VisitChangeRecord:
    """Visit ID change event (emitted once per detected change)."""

    t: datetime
    visit_id: int
    mode: str = "unknown"


@dataclass
class DesignChangeRecord:
    """Design ID change event (emitted once per detected change)."""

    t: datetime
    design_id: int


@dataclass
class ReconfigureRecord:
    """Autoguide parameter reconfigure (no visit transition).

    Emitted for ``autoguide reconfigure`` commands that change AG parameters
    (e.g. ``max_correction``, ``exposure_time``) without advancing the visit.
    """

    t: datetime
    params: dict
    visit_id: int | None = None
    mode: str = "unknown"


@dataclass
class GuideCatalogPerCamRecord:
    """Per-camera guide star counts, emitted once per call to ``get_guide_objects``.

    Attributes
    ----------
    t : datetime
        Timestamp (UTC) of the log line.
    n_catalog : tuple[int, ...]
        Raw guide star count per camera 1–6 (before any filtering).
    n_filtered : tuple[int, ...]
        Valid guide star count per camera 1–6 (after ``filter_guide_objects``).
    visit_id, frame_id : int | None
        Active PFS visit and AGC frame at emission time.
    mode : str
        Guiding mode active when this record was emitted.
    """

    t: datetime
    n_catalog: tuple
    n_filtered: tuple
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


@dataclass
class PerFrameCountRecord:
    """Per-camera object count funnel for a single AGC exposure.

    Attributes
    ----------
    t : datetime
        Timestamp (UTC) of the log line.
    n_detected_filtered : tuple[int, ...]
        Detected sources per camera after shape/quality filtering.
    n_matched : tuple[int, ...]
        Sources per camera that were paired with a catalog entry.
    n_valid : tuple[int, ...]
        Inlier-matched sources per camera after outlier rejection.
    visit_id, frame_id : int | None
        Active PFS visit and AGC frame at emission time.
    mode : str
        Guiding mode active when this record was emitted.
    """

    t: datetime
    n_detected_filtered: tuple
    n_matched: tuple
    n_valid: tuple
    visit_id: int | None = None
    frame_id: int | None = None
    mode: str = "unknown"


# ──────────────────────────────── Log Parsing ─────────────────────────────────

_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)Z")

_GUIDE_ERR = re.compile(
    r"guideErrors=(\d+)"
    r",([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)"
    r",([-\d.e+]+),([-\d.e+]+)"
    r",([-\d.e+]+),([-\d.e+]+)"
    r",(\w+)"
)
_STAR_STATS = re.compile(
    r"flux=np\.float64\(([-\d.e+]+)\)"
    r",peak=np\.float64\(([-\d.e+]+)\)"
    r",size=np\.float64\(([-\d.e+]+)\)"
)
_MATCHED = re.compile(r"Matched sources with valid residuals: (\d+)")
_CAM_COUNT = re.compile(r"AGC\[(\d)\]: find (\d+) objects")
# Per-camera count funnel — new log lines (data.py / FieldAcquisitionAndFocusing.py)
_CATALOG_PER_CAM = re.compile(r"guide_catalog_per_camera=([\d,]+)")
_GUIDE_FILT_PER_CAM = re.compile(r"guide_filtered_per_camera=([\d,]+)")
_DET_FILT_PER_CAM = re.compile(r"detected_filtered_per_camera=([\d,]+)")
_MATCHED_PER_CAM = re.compile(r"matched_per_camera=([\d,]+)")
_VALID_PER_CAM = re.compile(r"valid_matched_per_camera=([\d,]+)")
_TEL_AXES = re.compile(r"tel_axes=([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)")
_TEL_ROT = re.compile(r"tel_rot=([-\d.]+),([-\d.]+)")
_TEL_STATE = re.compile(
    r"data\.py:497 taken_at=[\d.]+,inr=([-\d.]+),adc=([-\d.]+),m2_pos3=([-\d.]+)"
)
_TEL_AXES_SK = re.compile(r"\[([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\]")
_FOCUS_Z0 = re.compile(r"'guide_delta_z': (nan|[-\d.e+]+)")
_FOCUS_ZN = re.compile(r"'guide_delta_z([1-6])': (nan|[-\d.e+]+)")
# Extracts visit_id and design_id from any command line
_VISIT_CMD = re.compile(r"\bvisit_id=(\d+)")
_DESIGN_CMD = re.compile(r"\bdesign_id=(\d+)")
# autoguide reconfigure uses visit= instead of visit_id=
_RECONFIGURE_VISIT = re.compile(r"new cmd: autoguide reconfigure.*?\bvisit=(\d+)")
# acquire_field with visit0= means a convergence-prep frame (always followed by autoguide start).
# Without visit0= (and without guide=no) it is a focus-sweep frame.
_ACQUIRE_VISIT0 = re.compile(r"\bvisit0=\d+")
# agcc frame-done acknowledgement
_FRAME_ID = re.compile(r'agc_frameid="(\d+)"')
# key=value pairs in autoguide reconfigure parameter-only commands
_KV = re.compile(r'\b([a-z][a-z0-9_]*)=([\w.+-]+)')


def _flt(s: str) -> float:
    return float("nan") if s == "nan" else float(s)


def _ts(line: str) -> Optional[datetime]:
    m = _LOG_TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)


def parse_line(line: str, state: dict) -> list[tuple[str, object]]:
    """Parse one raw log line into typed record objects.

    Parameters
    ----------
    line : str
        A single raw log line.
    state : dict
        Mutable session state. Tracks ``visit_id``, ``frame_id``, ``mode``,
        ``design_id``, and partial camera-count accumulation (``cams``).

    Returns
    -------
    list[tuple[str, object]]
        A (possibly empty) list of ``(record_type, record)`` tuples ready to
        push into a ``DataStore``.
    """

    t = _ts(line)
    if t is None:
        return []

    # ── Command lines: update context and emit visit/design change events ─────
    # Every new command can carry visit_id, design_id, and/or mode transitions.
    if "new cmd:" in line:
        if "acquire_field" in line:
            # guide=no → explicit focus-only visit (old style, May 2026-05-05 era).
            # No visit0= → focus sweep (new style: repeated frames, same visit_id, never
            #   transitions to autoguide start).  visit0= present → convergence-prep frame
            #   (single frame that immediately precedes autoguide start).
            if "guide=no" in line or not _ACQUIRE_VISIT0.search(line):
                state["mode"] = "focus"
            else:
                state["mode"] = "acquire"
        elif "autoguide start" in line:
            # autoguide start always opens a convergence visit (visit0 == visit_id)
            state["mode"] = "converge"
        elif "autoguide reconfigure" in line:
            # Only advance to 'autoguide' (science) mode when a new visit= is specified.
            # A reconfigure without visit= is a parameter-only change; mode stays as-is.
            if _RECONFIGURE_VISIT.search(line) or _VISIT_CMD.search(line):
                state["mode"] = "autoguide"
        elif "autoguide stop" in line:
            state["mode"] = "idle"

        # Most commands use visit_id=N; autoguide reconfigure uses visit=N
        vm = _VISIT_CMD.search(line)
        if vm:
            new_visit = int(vm.group(1))
        elif "autoguide reconfigure" in line:
            rm = _RECONFIGURE_VISIT.search(line)
            new_visit = int(rm.group(1)) if rm else None
        else:
            new_visit = None

        dm = _DESIGN_CMD.search(line)
        new_design = int(dm.group(1)) if dm else None

        results: list[tuple[str, object]] = []
        if new_visit is not None and new_visit != state.get("visit_id"):
            state["visit_id"] = new_visit
            results.append(("visit_changes", VisitChangeRecord(t=t, visit_id=new_visit, mode=state.get("mode", "unknown"))))
        if new_design is not None and new_design != state.get("design_id"):
            state["design_id"] = new_design
            results.append(("design_changes", DesignChangeRecord(t=t, design_id=new_design)))

        # Param-only reconfigure: no visit transition, but parameters changed
        if "autoguide reconfigure" in line and new_visit is None:
            # Strip the command prefix and collect all key=value pairs
            after_cmd = line.split("autoguide reconfigure", 1)[-1]
            params = {m.group(1): m.group(2) for m in _KV.finditer(after_cmd)}
            if params:
                results.append((
                    "reconfigs",
                    ReconfigureRecord(
                        t=t,
                        params=params,
                        visit_id=state.get("visit_id"),
                        mode=state.get("mode", "unknown"),
                    ),
                ))

        return results

    # ── Frame ID: agcc reports the new frame ID before data records arrive ────
    if 'agc_frameid="' in line:
        fm = _FRAME_ID.search(line)
        if fm:
            state["frame_id"] = int(fm.group(1))
        return []

    # ── Per-exposure data records ─────────────────────────────────────────────
    visit_id = state.get("visit_id")
    frame_id = state.get("frame_id")
    mode = state.get("mode", "unknown")

    # Guide solution — also acts as the authoritative frame_id source
    m = _GUIDE_ERR.search(line)
    if m:
        fid = int(m.group(1))
        state["frame_id"] = fid
        return [
            (
                "guide",
                GuideRecord(
                    t=t,
                    frame_id=fid,
                    dra=float(m.group(2)),
                    ddec=float(m.group(3)),
                    dinr=float(m.group(4)),
                    daz=float(m.group(5)),
                    del_=float(m.group(6)),
                    dfocus=float(m.group(7)),
                    dscale=float(m.group(8)),
                    status=m.group(9),
                    visit_id=visit_id,
                    mode=mode,
                ),
            )
        ]

    # Per-camera focus (from agc_guide_offsets)
    if "guide_delta_z1" in line:
        zm = _FOCUS_Z0.search(line)
        z = _flt(zm.group(1)) if zm else float("nan")
        zv = {int(m.group(1)): _flt(m.group(2)) for m in _FOCUS_ZN.finditer(line)}
        return [
            (
                "focus",
                FocusRecord(
                    t=t,
                    z=z,
                    z1=zv.get(1, float("nan")),
                    z2=zv.get(2, float("nan")),
                    z3=zv.get(3, float("nan")),
                    z4=zv.get(4, float("nan")),
                    z5=zv.get(5, float("nan")),
                    z6=zv.get(6, float("nan")),
                    visit_id=visit_id,
                    frame_id=frame_id,
                    mode=mode,
                ),
            )
        ]

    # Star stats
    m = _STAR_STATS.search(line)
    if m:
        return [
            (
                "star_stats",
                StarStatsRecord(
                    t=t,
                    flux=float(m.group(1)),
                    peak=float(m.group(2)),
                    size=float(m.group(3)),
                    visit_id=visit_id,
                    frame_id=frame_id,
                    mode=mode,
                ),
            )
        ]

    # Matched count
    m = _MATCHED.search(line)
    if m:
        return [
            (
                "matched",
                MatchedRecord(t=t, n=int(m.group(1)), visit_id=visit_id, frame_id=frame_id, mode=mode),
            )
        ]

    # Per-camera counts across AGC[1]..AGC[6]
    m = _CAM_COUNT.search(line)
    if m:
        cam, n = int(m.group(1)), int(m.group(2))
        cams: dict = state.setdefault("cams", {})
        if cam == 1 and cams:
            cams.clear()  # flush incomplete set from previous exposure
        cams[cam] = n
        if cam == 6:
            rec = CameraCountRecord(
                t=t,
                counts=tuple(cams.get(i) for i in range(1, 7)),
                visit_id=visit_id,
                frame_id=frame_id,
                mode=mode,
            )
            cams.clear()
            return [("camera_counts", rec)]
        return []

    # Per-camera guide catalog counts — guide_catalog_per_camera / guide_filtered_per_camera
    m = _CATALOG_PER_CAM.search(line)
    if m:
        state.setdefault("cam_cat_acc", {})["n_catalog"] = tuple(int(x) for x in m.group(1).split(","))
        return []

    m = _GUIDE_FILT_PER_CAM.search(line)
    if m:
        acc = state.setdefault("cam_cat_acc", {})
        acc["n_filtered"] = tuple(int(x) for x in m.group(1).split(","))
        if "n_catalog" in acc:
            rec = GuideCatalogPerCamRecord(
                t=t,
                n_catalog=acc.pop("n_catalog"),
                n_filtered=acc.pop("n_filtered"),
                visit_id=visit_id,
                frame_id=frame_id,
                mode=mode,
            )
            return [("guide_catalog_counts", rec)]
        return []

    # Per-camera per-frame detection/match funnel
    m = _DET_FILT_PER_CAM.search(line)
    if m:
        state.setdefault("cam_frame_acc", {})["n_det_filt"] = tuple(int(x) for x in m.group(1).split(","))
        return []

    m = _VALID_PER_CAM.search(line)
    if m:
        acc = state.setdefault("cam_frame_acc", {})
        acc["n_valid"] = tuple(int(x) for x in m.group(1).split(","))
        if "n_det_filt" in acc and "n_matched" in acc:
            rec = PerFrameCountRecord(
                t=t,
                n_detected_filtered=acc.pop("n_det_filt"),
                n_matched=acc.pop("n_matched"),
                n_valid=acc.pop("n_valid"),
                visit_id=visit_id,
                frame_id=frame_id,
                mode=mode,
            )
            return [("per_frame_counts", rec)]
        return []

    m = _MATCHED_PER_CAM.search(line)
    if m:
        state.setdefault("cam_frame_acc", {})["n_matched"] = tuple(int(x) for x in m.group(1).split(","))
        return []

    # Telescope axes (reply lines) — Az, El, parallactic rot (group 4/airmass dropped)
    if "tel_axes=" in line and "reply=" in line:
        m = _TEL_AXES.search(line)
        if m:
            return [
                (
                    "tel_axes",
                    TelAxesRecord(
                        t=t,
                        az=float(m.group(1)),
                        el=float(m.group(2)),
                        rot=float(m.group(3)),
                    ),
                )
            ]

    # Instrument rotator (reply lines) — sky PA + physical insrot angle, high-cadence
    if "tel_rot=" in line and "reply=" in line:
        m = _TEL_ROT.search(line)
        if m:
            return [("tel_rot", TelRotRecord(t=t, pa=float(m.group(1)), insrot=float(m.group(2))))]

    # Telescope axes (receiveStatusKeys)
    if "receiveStatusKeys: gen2,tel_axes" in line:
        m = _TEL_AXES_SK.search(line)
        if m:
            return [
                (
                    "tel_axes",
                    TelAxesRecord(
                        t=t,
                        az=float(m.group(1)),
                        el=float(m.group(2)),
                        rot=float(m.group(3)),
                    ),
                )
            ]

    # Per-frame telescope state: InR, ADC, M2 position
    if "data.py:497" in line:
        m = _TEL_STATE.search(line)
        if m:
            return [
                (
                    "tel_state",
                    TelStateRecord(
                        t=t,
                        inr=float(m.group(1)),
                        adc=float(m.group(2)),
                        m2_pos3=float(m.group(3)),
                        visit_id=state.get("visit_id"),
                        frame_id=state.get("frame_id"),
                        mode=state.get("mode", "unknown"),
                    ),
                )
            ]

    return []


# ──────────────────────────────── Data Storage ────────────────────────────────

class DataStore:
    """Thread-safe rolling buffer of parsed records.

    Parameters
    ----------
    window_minutes : int | None, default 30
        Rolling time window in minutes. If ``None``, records are never pruned.
    """

    def __init__(self, window_minutes: Optional[int] = 30) -> None:
        self._window = timedelta(minutes=window_minutes) if window_minutes is not None else None
        self._lock = threading.Lock()
        self.guide: deque = deque()
        self.focus: deque = deque()
        self.star_stats: deque = deque()
        self.matched: deque = deque()
        self.camera_counts: deque = deque()
        self.tel_axes: deque = deque()
        self.tel_rot: deque = deque()
        self.tel_state: deque = deque()
        self.visit_changes: deque = deque()
        self.design_changes: deque = deque()
        self.reconfigs: deque = deque()
        self.guide_catalog_counts: deque = deque()
        self.per_frame_counts: deque = deque()

    def push(self, rec_type: str, rec) -> None:
        with self._lock:
            if not hasattr(self, rec_type):
                setattr(self, rec_type, deque())
            q: deque = getattr(self, rec_type)
            q.append(rec)
            if self._window is not None:
                cutoff = rec.t - self._window
                while q and q[0].t < cutoff:
                    q.popleft()

    def snapshot(self, name: str) -> list:
        """Return a list copy of a deque (safe without holding the lock)."""
        with self._lock:
            return list(getattr(self, name, deque()))

    _ALL_QUEUES = (
        "guide",
        "focus",
        "star_stats",
        "matched",
        "camera_counts",
        "tel_axes",
        "tel_rot",
        "tel_state",
        "visit_changes",
        "design_changes",
        "reconfigs",
        "guide_catalog_counts",
        "per_frame_counts",
    )

    def slice(self, t_start=None, t_end=None) -> "DataStore":
        """Return a new DataStore containing only records in ``[t_start, t_end)``."""
        new = DataStore(window_minutes=None)
        with self._lock:
            for attr in self._ALL_QUEUES:
                for rec in getattr(self, attr):
                    if (t_start is None or rec.t >= t_start) and (t_end is None or rec.t < t_end):
                        new.push(attr, rec)
        return new


# ─────────────────────────────── Night boundaries ─────────────────────────────

def _night_bounds(ref: datetime) -> tuple[datetime, datetime]:
    """Return (start, end) in UTC for the HST observing night containing `ref`.

    The night runs from 18:00 HST to 06:00 HST the following morning.
    """

    hst = ref.astimezone(HST)
    if hst.hour < 6:
        # After midnight — night started the previous evening
        evening = (hst - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        evening = hst.replace(hour=18, minute=0, second=0, microsecond=0)
    morning = evening + timedelta(hours=12)
    return evening.astimezone(timezone.utc), morning.astimezone(timezone.utc)


def _night_bounds_from_store(store: DataStore) -> tuple[datetime, datetime]:
    """Determine tonight's x-axis limits from the earliest record in `store`, or now()."""

    for attr in ("guide", "tel_axes", "tel_state", "star_stats", "camera_counts"):
        recs = store.snapshot(attr)
        if recs:
            return _night_bounds(recs[0].t)
    return _night_bounds(datetime.now(timezone.utc))
