#!/usr/bin/env python3
"""
ag_dashboard.py — Live-updating dashboard for PFS AG actor logs.

Usage:
    python ag_dashboard.py                           # tail local current.log
    python ag_dashboard.py --path /data/logs/ag.log
    python ag_dashboard.py --source ssh-subprocess --host pfsa-ics01 --path /data/logs/current.log
    python ag_dashboard.py --source paramiko       --host pfsa-ics01 --path /data/logs/current.log
    python ag_dashboard.py --replay                  # read from start of file (testing)

Panels:
    1. Guide offsets (dRA, dDec, dInR, dAz, dEl) vs time
    2. Focus offsets (combined + per-camera) and dScale
    3. Star quality: PSF size, flux, matched star count
    4. Per-camera object counts (AGC 1-6)
    5. Telescope Az / El / Airmass
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HST = timezone(timedelta(hours=-10))  # Hawaii Standard Time (no DST)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Records
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuideRecord:
    """From guideErrors= line. Per-exposure guide solution."""
    t: datetime
    exp_id: int
    dra: float    # arcsec
    ddec: float   # arcsec
    dinr: float   # arcsec  (rotation)
    daz: float    # arcsec
    del_: float   # arcsec  (elevation)
    dfocus: float # mm
    dscale: float # dimensionless


@dataclass
class FocusRecord:
    """Per-camera focus offsets from agc_guide_offsets write line (mm)."""
    t: datetime
    z: float    # combined
    z1: float
    z2: float
    z3: float
    z4: float
    z5: float
    z6: float


@dataclass
class StarStatsRecord:
    """From 'Calculated star stats' line."""
    t: datetime
    flux: float  # ADU
    peak: float  # ADU
    size: float  # PSF FWHM proxy, pixels


@dataclass
class MatchedRecord:
    """From 'Matched sources with valid residuals' line."""
    t: datetime
    n: int


@dataclass
class CameraCountRecord:
    """Detected sources per camera per frame. counts[i] is None if camera absent."""
    t: datetime
    counts: tuple  # length 6, each int or None


@dataclass
class TelAxesRecord:
    """From tel_axes= (reply lines) or receiveStatusKeys."""
    t: datetime
    az: float
    el: float
    rot: float
    airmass: float


@dataclass
class VisitChangeRecord:
    """Emitted once when visit_id changes (from 'new cmd: acquire_field')."""
    t: datetime
    visit_id: int


@dataclass
class DesignChangeRecord:
    """Emitted once when design_id changes (from 'new cmd: acquire_field')."""
    t: datetime
    design_id: int


# ═══════════════════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════════════════

_LOG_TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)Z')

_GUIDE_ERR = re.compile(
    r'guideErrors=(\d+)'
    r',([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)'
    r',([-\d.e+]+),([-\d.e+]+)'
    r',([-\d.e+]+),([-\d.e+]+)'
    r',(OK|ERROR)'
)
_STAR_STATS = re.compile(
    r'flux=np\.float64\(([-\d.e+]+)\)'
    r',peak=np\.float64\(([-\d.e+]+)\)'
    r',size=np\.float64\(([-\d.e+]+)\)'
)
_MATCHED    = re.compile(r'Matched sources with valid residuals: (\d+)')
_CAM_COUNT  = re.compile(r'AGC\[(\d)\]: find (\d+) objects')
_TEL_AXES   = re.compile(r'tel_axes=([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)')
_TEL_AXES_SK = re.compile(r'\[([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\]')
_FOCUS_Z0   = re.compile(r"'guide_delta_z': (nan|[-\d.e+]+)")
_FOCUS_ZN   = re.compile(r"'guide_delta_z([1-6])': (nan|[-\d.e+]+)")
_ACQUIRE    = re.compile(r'new cmd: acquire_field.*?design_id=(\d+).*?visit_id=(\d+)')
_RECONFIGURE = re.compile(r'new cmd: autoguide reconfigure.*?\bvisit=(\d+)')


def _flt(s: str) -> float:
    return float('nan') if s == 'nan' else float(s)


def _ts(line: str) -> Optional[datetime]:
    m = _LOG_TS.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)


def parse_line(line: str, state: dict) -> list[tuple[str, object]]:
    """
    Parse one raw log line.

    *state* is a mutable dict used for cross-line accumulation (e.g. building up
    per-camera counts across six successive AGC lines).  Pass the same dict for
    every line in a session.  Returns a (possibly empty) list of
    ``(record_type, record)`` tuples ready to push into the DataStore.
    """
    t = _ts(line)
    if t is None:
        return []

    # ── Guide solution (one per exposure) ─────────────────────────────────────
    m = _GUIDE_ERR.search(line)
    if m:
        return [('guide', GuideRecord(
            t=t, exp_id=int(m.group(1)),
            dra=float(m.group(2)),  ddec=float(m.group(3)),
            dinr=float(m.group(4)), daz=float(m.group(5)),  del_=float(m.group(6)),
            dfocus=float(m.group(7)), dscale=float(m.group(8)),
        ))]

    # ── Per-camera focus (from agc_guide_offsets write line) ──────────────────
    if 'guide_delta_z1' in line:
        zm = _FOCUS_Z0.search(line)
        z  = _flt(zm.group(1)) if zm else float('nan')
        zv = {int(m.group(1)): _flt(m.group(2)) for m in _FOCUS_ZN.finditer(line)}
        return [('focus', FocusRecord(
            t=t, z=z,
            z1=zv.get(1, float('nan')), z2=zv.get(2, float('nan')),
            z3=zv.get(3, float('nan')), z4=zv.get(4, float('nan')),
            z5=zv.get(5, float('nan')), z6=zv.get(6, float('nan')),
        ))]

    # ── Star stats ────────────────────────────────────────────────────────────
    m = _STAR_STATS.search(line)
    if m:
        return [('star_stats', StarStatsRecord(
            t=t, flux=float(m.group(1)), peak=float(m.group(2)), size=float(m.group(3)),
        ))]

    # ── Matched guide star count ───────────────────────────────────────────────
    m = _MATCHED.search(line)
    if m:
        return [('matched', MatchedRecord(t=t, n=int(m.group(1))))]

    # ── Per-camera counts (accumulate across AGC[1]..AGC[6] lines) ────────────
    m = _CAM_COUNT.search(line)
    if m:
        cam, n = int(m.group(1)), int(m.group(2))
        cams: dict = state.setdefault('cams', {})
        if cam == 1 and cams:
            cams.clear()            # flush incomplete set from previous exposure
        cams[cam] = n
        if cam == 6:
            rec = CameraCountRecord(t=t, counts=tuple(cams.get(i) for i in range(1, 7)))
            cams.clear()
            return [('camera_counts', rec)]
        return []

    # ── Telescope axes (from reply= lines, per-exposure frequency) ────────────
    if 'tel_axes=' in line and 'reply=' in line:
        m = _TEL_AXES.search(line)
        if m:
            return [('tel_axes', TelAxesRecord(
                t=t,
                az=float(m.group(1)), el=float(m.group(2)),
                rot=float(m.group(3)), airmass=float(m.group(4)),
            ))]

    # ── Telescope axes (from receiveStatusKeys, between-observation frequency) ─
    if 'receiveStatusKeys: gen2,tel_axes' in line:
        m = _TEL_AXES_SK.search(line)
        if m:
            return [('tel_axes', TelAxesRecord(
                t=t,
                az=float(m.group(1)), el=float(m.group(2)),
                rot=float(m.group(3)), airmass=float(m.group(4)),
            ))]

    # ── Visit / design ID changes ─────────────────────────────────────────────
    if 'new cmd: acquire_field' in line:
        m = _ACQUIRE.search(line)
        if m:
            design_id, visit_id = int(m.group(1)), int(m.group(2))
            results = []
            if visit_id != state.get('last_visit_id'):
                state['last_visit_id'] = visit_id
                results.append(('visit_changes', VisitChangeRecord(t=t, visit_id=visit_id)))
            if design_id != state.get('last_design_id'):
                state['last_design_id'] = design_id
                results.append(('design_changes', DesignChangeRecord(t=t, design_id=design_id)))
            return results

    if 'new cmd: autoguide reconfigure' in line:
        m = _RECONFIGURE.search(line)
        if m:
            visit_id = int(m.group(1))
            if visit_id != state.get('last_visit_id'):
                state['last_visit_id'] = visit_id
                return [('visit_changes', VisitChangeRecord(t=t, visit_id=visit_id))]

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Data Store
# ═══════════════════════════════════════════════════════════════════════════════

class DataStore:
    """
    Thread-safe rolling buffer.

    If *window_minutes* is None, records are never pruned (use with --mode full).
    Otherwise records older than the window are dropped as new ones arrive.
    """

    def __init__(self, window_minutes: Optional[int] = 30) -> None:
        self._window = timedelta(minutes=window_minutes) if window_minutes is not None else None
        self._lock   = threading.Lock()
        self.guide:          deque = deque()
        self.focus:          deque = deque()
        self.star_stats:     deque = deque()
        self.matched:        deque = deque()
        self.camera_counts:  deque = deque()
        self.tel_axes:       deque = deque()
        self.visit_changes:  deque = deque()
        self.design_changes: deque = deque()

    def push(self, rec_type: str, rec) -> None:
        with self._lock:
            q: deque = getattr(self, rec_type)
            q.append(rec)
            if self._window is not None:
                cutoff = rec.t - self._window
                while q and q[0].t < cutoff:
                    q.popleft()

    def snapshot(self, name: str) -> list:
        """Return a list copy of a deque (safe to read without holding the lock)."""
        with self._lock:
            return list(getattr(self, name))

    _ALL_QUEUES = ('guide', 'focus', 'star_stats', 'matched',
                   'camera_counts', 'tel_axes', 'visit_changes', 'design_changes')

    def slice(self, t_start=None, t_end=None) -> 'DataStore':
        """Return a new DataStore containing only records in [t_start, t_end)."""
        new = DataStore(window_minutes=None)
        with self._lock:
            for attr in self._ALL_QUEUES:
                for rec in getattr(self, attr):
                    if (t_start is None or rec.t >= t_start) and \
                       (t_end   is None or rec.t <  t_end):
                        new.push(attr, rec)
        return new


# ═══════════════════════════════════════════════════════════════════════════════
# Log Sources
# ═══════════════════════════════════════════════════════════════════════════════

class LogSource(ABC):
    @abstractmethod
    def lines(self):
        """Yield raw log lines indefinitely (blocking generator)."""
        ...


class LocalFileTailSource(LogSource):
    """Poll a local file; seek to end unless *from_start* is True."""

    def __init__(self, path: str, from_start: bool = False) -> None:
        self.path       = path
        self.from_start = from_start

    def lines(self):
        with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
            if not self.from_start:
                f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip('\n')
                else:
                    time.sleep(0.05)


class SSHSubprocessSource(LogSource):
    """Stream via ``ssh host tail -F path`` using a subprocess."""

    def __init__(self, host: str, path: str) -> None:
        self.host = host
        self.path = path

    def lines(self):
        import subprocess
        cmd  = ['ssh', self.host, f'tail -F {self.path!r}']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                yield line.rstrip('\n')
        finally:
            proc.terminate()
            proc.wait()


class SSHParamikoSource(LogSource):
    """Stream via paramiko (programmatic SSH; easier reconnect handling)."""

    def __init__(self, host: str, path: str,
                 username: Optional[str] = None, port: int = 22) -> None:
        self.host     = host
        self.path     = path
        self.username = username
        self.port     = port

    def lines(self):
        import paramiko
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, port=self.port, username=self.username)
        try:
            _, stdout, _ = client.exec_command(f'tail -F {self.path!r}')
            for line in stdout:
                yield line.rstrip('\n')
        finally:
            client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard — abstract base + panels + matplotlib implementation
# ═══════════════════════════════════════════════════════════════════════════════

def _night_bounds(ref: datetime) -> tuple[datetime, datetime]:
    """
    Return (start, end) in UTC for the HST observing night that contains *ref*.
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
    """Determine tonight's x-axis limits from the earliest record in store, or from now()."""
    for attr in ('guide', 'tel_axes', 'star_stats', 'camera_counts'):
        recs = store.snapshot(attr)
        if recs:
            return _night_bounds(recs[0].t)
    return _night_bounds(datetime.now(timezone.utc))


def _init_date_axis(ax) -> None:
    """Attach a HST-aware ConciseDateFormatter to *ax*."""
    loc = mdates.AutoDateLocator(tz=HST)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=HST))


class Panel(ABC):
    """Abstract base for one subplot row in the dashboard."""

    needs_twin: bool = False   # set True if a right-hand y-axis is required

    @abstractmethod
    def setup(self, ax, tax=None) -> None:
        """Called once: create Line2D objects, set labels and legend."""
        ...

    @abstractmethod
    def update(self, ax, tax, store: DataStore) -> None:
        """Called every animation frame: refresh line data."""
        ...

    @staticmethod
    def _set_window(ax, recs, store: DataStore) -> None:
        """Pin x-axis to the rolling window. No-op when window is unbounded."""
        if not recs or store._window is None:
            return
        t_max = recs[-1].t
        ax.set_xlim(t_max - store._window, t_max)

    def draw_event_lines(self, ax, store: DataStore, *, with_labels: bool = False) -> None:
        """
        Draw vertical lines for visit and design ID changes on *ax*.

        Lines are drawn on every panel so events can be cross-referenced.
        Labels are only drawn when *with_labels* is True (typically the bottom panel).

        Visit changes : solid royal-blue line, labelled 'v<id>'.
        Design changes: dashed orange line, labelled 'd<last-8-digits>'.
        """
        if not hasattr(self, '_drawn_visits'):
            self._drawn_visits:  set = set()
            self._drawn_designs: set = set()

        for rec in store.snapshot('visit_changes'):
            if rec.t not in self._drawn_visits:
                ax.axvline(rec.t, color='royalblue', lw=1.4, ls='-', alpha=0.3, zorder=2)
                if with_labels:
                    # ha='left': text starts to the right of the line
                    ax.text(rec.t, 0.99, f' v{rec.visit_id}',
                            color='royalblue', fontsize=7, va='top', ha='left', rotation=90,
                            transform=ax.get_xaxis_transform(), clip_on=True)
                self._drawn_visits.add(rec.t)

        for rec in store.snapshot('design_changes'):
            if rec.t not in self._drawn_designs:
                ax.axvline(rec.t, color='darkorange', lw=1.4, ls='--', alpha=0.3, zorder=2)
                if with_labels:
                    # ha='right': text ends to the left of the line (label reads above/before)
                    ax.text(rec.t, 0.99, f'd{str(rec.design_id)[-8:]} ',
                            color='darkorange', fontsize=7, va='top', ha='right', rotation=90,
                            transform=ax.get_xaxis_transform(), clip_on=True)
                self._drawn_designs.add(rec.t)


# ── Panel 1: guide offsets ────────────────────────────────────────────────────

class GuideOffsetsPanel(Panel):
    """
    dRA, dDec, dAz, dEl on the left y-axis (arcsec, small corrections).
    dInR on the right y-axis (arcsec, but typically much larger rotation values).
    """

    needs_twin = True

    _LEFT_FIELDS  = [('dra', 'dRA'), ('ddec', 'dDec'), ('daz', 'dAz'), ('del_', 'dEl')]
    _LEFT_COLORS  = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    def setup(self, ax, tax=None):
        self._left_lines = {}
        for (attr, label), color in zip(self._LEFT_FIELDS, self._LEFT_COLORS):
            (ln,) = ax.plot([], [], label=label, color=color, lw=1.2, marker='.', ms=4)
            self._left_lines[attr] = ln
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_ylabel('arcsec')
        ax.set_title('Guide offsets')
        _init_date_axis(ax)

        if tax:
            (self._inr_ln,) = tax.plot([], [], label='dInR', color='#2ca02c',
                                        lw=1.2, ls='--', marker='.', ms=4)
            tax.set_ylabel('dInR (arcsec)', color='#2ca02c')
            tax.tick_params(axis='y', labelcolor='#2ca02c')

        handles = list(self._left_lines.values()) + ([self._inr_ln] if tax else [])
        ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=5)

    def update(self, ax, tax, store):
        recs = store.snapshot('guide')
        if not recs:
            return
        ts = [r.t for r in recs]
        for attr, ln in self._left_lines.items():
            ln.set_xdata(ts)
            ln.set_ydata([getattr(r, attr) for r in recs])
        ax.relim()
        ax.autoscale_view(scalex=False)
        if tax:
            self._inr_ln.set_xdata(ts)
            self._inr_ln.set_ydata([r.dinr for r in recs])
            tax.relim()
            tax.autoscale_view(scalex=False)
        self._set_window(ax, recs, store)


# ── Panel 2: focus & scale ────────────────────────────────────────────────────

class FocusPanel(Panel):
    """Combined dFocus + per-camera dZ1..dZ6 (mm, left) and dScale (right).

    All data shown as markers only (no connecting lines) so individual exposures
    are readable without visual clutter. dFocus has a larger marker for emphasis.
    Z3 legend entry updates to "(offline)" in grey when all Z3 values are NaN.
    """

    needs_twin = True
    _Z_FIELDS  = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6']
    # Z6 changed from brownish #8c564b (reads as grey on many monitors) to teal
    _Z_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    _Z_LABELS  = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']

    def setup(self, ax, tax=None):
        (self._focus_ln,) = ax.plot([], [], label='dFocus', color='black',
                                    lw=1.0, marker='.', ms=6, zorder=3)
        self._z_lines: dict = {}
        for attr, color, label in zip(self._Z_FIELDS, self._Z_COLORS, self._Z_LABELS):
            (ln,) = ax.plot([], [], label=label, color=color, lw=0, marker='.', ms=4)
            self._z_lines[attr] = ln
        ax.set_ylabel('Focus offset (mm)')
        ax.set_title('Focus offsets & scale')
        _init_date_axis(ax)
        ax.axhline(0, color='k', lw=0.8, alpha=0.25, zorder=0)
        self._z3_offline_state = None  # track to avoid rebuilding legend every frame
        self._tax = tax  # store for legend rebuilds

        if tax:
            (self._scale_ln,) = tax.plot([], [], color='gray', lw=1.0, marker='.', ms=4, label='dScale ×10⁻⁶')
            tax.set_ylabel('dScale (×10⁻⁶)', color='gray')
            tax.tick_params(axis='y', labelcolor='gray')

        self._rebuild_legend(ax)

    def _rebuild_legend(self, ax):
        """Build legend from both ax and tax artists so dScale is always included."""
        handles = [self._focus_ln] + list(self._z_lines.values())
        if self._tax:
            handles.append(self._scale_ln)
        ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=8)

    def update(self, ax, tax, store):
        import math
        g = store.snapshot('guide')
        if g:
            ts = [r.t for r in g]
            self._focus_ln.set_xdata(ts)
            self._focus_ln.set_ydata([r.dfocus for r in g])
            if tax:
                self._scale_ln.set_xdata(ts)
                self._scale_ln.set_ydata([r.dscale * 1e6 for r in g])
                tax.relim()
                tax.autoscale_view(scalex=False)

        f = store.snapshot('focus')
        if f:
            ts = [r.t for r in f]
            for attr, ln in self._z_lines.items():
                ln.set_xdata(ts)
                ln.set_ydata([getattr(r, attr) for r in f])
            z3_offline = all(math.isnan(r.z3) for r in f)
            if z3_offline != self._z3_offline_state:
                self._z3_offline_state = z3_offline
                z3_ln = self._z_lines['z3']
                if z3_offline:
                    z3_ln.set_color('lightgrey')
                    z3_ln.set_label('Z3 (offline)')
                else:
                    z3_ln.set_color('#2ca02c')
                    z3_ln.set_label('Z3')
                self._rebuild_legend(ax)
        ax.relim()
        ax.autoscale_view(scalex=False)
        self._set_window(ax, g or f, store)


# ── Panel 3: star quality ─────────────────────────────────────────────────────

class StarQualityPanel(Panel):
    """PSF size / FWHM (left, px) and guide star peak ADU on log scale (right).

    Peak ADU spikes within a visit typically indicate satellite/aircraft/cosmic-ray
    contamination in one of the guide cameras — not real guide star brightness changes.
    The log scale keeps the nominal level visible alongside the transient spikes.
    """

    needs_twin = True

    def setup(self, ax, tax=None):
        (self._size_ln,) = ax.plot([], [], label='PSF size (px)', color='#1f77b4', lw=1.2, marker='.', ms=4)
        ax.set_ylabel('PSF size (px)', color='#1f77b4')
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax.set_title('Star quality / seeing proxy')
        _init_date_axis(ax)

        if tax:
            (self._peak_ln,) = tax.plot([], [], color='#ff7f0e', lw=1.2, ls='--', marker='.', ms=4, label='peak (ADU, log)')
            tax.set_yscale('log')
            tax.set_ylabel('peak ADU (log)', color='#ff7f0e')
            tax.tick_params(axis='y', labelcolor='#ff7f0e')

        handles = [self._size_ln] + ([self._peak_ln] if tax else [])
        ax.legend(handles=handles, loc='upper left', fontsize=8)

    def update(self, ax, tax, store):
        ss = store.snapshot('star_stats')
        if ss:
            ts = [r.t for r in ss]
            self._size_ln.set_xdata(ts)
            self._size_ln.set_ydata([r.size for r in ss])
            ax.relim()
            ax.autoscale_view(scalex=False)
            if tax:
                self._peak_ln.set_xdata(ts)
                self._peak_ln.set_ydata([r.peak for r in ss])
                tax.relim()
                tax.autoscale_view(scalex=False)

        self._set_window(ax, ss, store)


# ── Panel 4: per-camera counts ────────────────────────────────────────────────

class CameraCountsPanel(Panel):
    """Detected sources per AG camera (1–6)."""

    _COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def setup(self, ax, tax=None):
        self._lines = []
        for i, color in enumerate(self._COLORS):
            (ln,) = ax.plot([], [], label=f'AGC{i + 1}', color=color,
                            lw=1, marker='.', ms=3)
            self._lines.append(ln)
        ax.set_ylabel('detected sources')
        ax.set_title('Per-camera object counts')
        ax.legend(loc='upper left', fontsize=8, ncol=6)
        _init_date_axis(ax)

    def update(self, ax, tax, store):
        recs = store.snapshot('camera_counts')
        if not recs:
            return
        ts = [r.t for r in recs]
        for i, ln in enumerate(self._lines):
            vals = [r.counts[i] if r.counts[i] is not None else float('nan')
                    for r in recs]
            ln.set_xdata(ts)
            ln.set_ydata(vals)
        ax.relim()
        ax.autoscale_view(scalex=False)
        self._set_window(ax, recs, store)


# ── Panel 5: telescope position ───────────────────────────────────────────────

class TelescopePanel(Panel):
    """Az and El (left, deg) and Airmass (right)."""

    needs_twin = True

    def setup(self, ax, tax=None):
        (self._az_ln,) = ax.plot([], [], label='Az (deg)', color='#1f77b4', lw=1.2, marker='.', ms=4)
        (self._el_ln,) = ax.plot([], [], label='El (deg)', color='#ff7f0e', lw=1.2, marker='.', ms=4)
        ax.set_ylabel('Az / El (deg)')
        ax.set_title('Telescope position')
        _init_date_axis(ax)

        if tax:
            (self._am_ln,) = tax.plot([], [], label='Airmass', color='#2ca02c',
                                       lw=1.2, ls='--', marker='.', ms=4)
            tax.set_ylabel('Airmass', color='#2ca02c')
            tax.tick_params(axis='y', labelcolor='#2ca02c')

        handles = [self._az_ln, self._el_ln] + ([self._am_ln] if tax else [])
        ax.legend(handles=handles, loc='upper left', fontsize=8)

    def update(self, ax, tax, store):
        recs = store.snapshot('tel_axes')
        if not recs:
            return
        ts = [r.t for r in recs]
        self._az_ln.set_xdata(ts); self._az_ln.set_ydata([r.az      for r in recs])
        self._el_ln.set_xdata(ts); self._el_ln.set_ydata([r.el      for r in recs])
        ax.relim()
        ax.autoscale_view(scalex=False)
        if tax:
            self._am_ln.set_xdata(ts)
            self._am_ln.set_ydata([r.airmass for r in recs])
            tax.relim()
            tax.autoscale_view(scalex=False)
        self._set_window(ax, recs, store)


# ── Dashboard ABC and matplotlib implementation ───────────────────────────────

class Dashboard(ABC):
    """
    Swap this out to change the plotting backend.
    Implement ``run(store, panels)`` to display the data however you like.
    The store and panel list are fully independent of this class.
    """

    @abstractmethod
    def run(self, store: DataStore, panels: list[Panel]) -> None:
        """Start the dashboard. Should block until the user closes it."""
        ...


class MatplotlibDashboard(Dashboard):
    """Renders panels as vertically stacked subplots using FuncAnimation."""

    def __init__(self, interval_ms: int = 2000) -> None:
        self.interval_ms = interval_ms

    def run(self, store: DataStore, panels: list[Panel]) -> None:
        n    = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle('PFS AG Actor Dashboard', fontsize=13, fontweight='bold')

        # Only show x-tick labels on the bottom panel
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        twin_axes: list = []
        for ax, panel in zip(axes, panels):
            tax = ax.twinx() if panel.needs_twin else None
            twin_axes.append(tax)
            panel.setup(ax, tax)

        plt.tight_layout()

        def _update(_frame):
            last_idx = len(panels) - 1
            for i, (ax, tax, panel) in enumerate(zip(axes, twin_axes, panels)):
                panel.update(ax, tax, store)
                panel.draw_event_lines(ax, store, with_labels=(i == last_idx))
            # Fix x-axis to tonight's HST night window across all shared panels.
            # This must be the last xlim call so it overrides any per-panel autoscaling.
            x_start, x_end = _night_bounds_from_store(store)
            axes[0].set_xlim(x_start, x_end)
            return []

        # Keep a reference to prevent garbage collection before plt.show() returns.
        self._ani = FuncAnimation(
            fig, _update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion thread
# ═══════════════════════════════════════════════════════════════════════════════

def _ingest(source: LogSource, store: DataStore, stop: threading.Event) -> None:
    """Read lines from *source*, parse them, and push records into *store*."""
    state: dict = {}
    while not stop.is_set():
        try:
            for line in source.lines():
                if stop.is_set():
                    return
                for rec_type, rec in parse_line(line, state):
                    store.push(rec_type, rec)
        except Exception as exc:
            if not stop.is_set():
                print(f'[ag_dashboard] ingest error: {exc!r} — retrying in 5 s')
                time.sleep(5)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Live-updating dashboard for PFS AG actor logs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input source ──────────────────────────────────────────────────────────
    ap.add_argument('--source', choices=['local', 'ssh-subprocess', 'paramiko'],
                    default='local', help='Log input method')
    ap.add_argument('--path', default='logs/current.log',
                    help='Path to the log file (local or remote)')
    ap.add_argument('--host', help='SSH hostname (ssh-subprocess / paramiko)')
    ap.add_argument('--user', help='SSH username  (paramiko only)')
    ap.add_argument('--port', type=int, default=22, help='SSH port (paramiko only)')

    # ── Local-source mode (mutually exclusive) ────────────────────────────────
    local_grp = ap.add_mutually_exclusive_group()
    local_grp.add_argument(
        '--tail', dest='local_mode', action='store_const', const='tail',
        help='(local, default) Follow the end of the file as new data arrives')
    local_grp.add_argument(
        '--replay', dest='local_mode', action='store_const', const='replay',
        help='(local) Read from the start of the file, show a rolling window')
    local_grp.add_argument(
        '--full', dest='local_mode', action='store_const', const='full',
        help='(local) Read the entire file and display all data — ignores --window')
    ap.set_defaults(local_mode='tail')

    # ── Display ───────────────────────────────────────────────────────────────
    ap.add_argument('--window', type=int, default=30, metavar='MIN',
                    help='Rolling window to display in minutes (--tail / --replay only)')
    ap.add_argument('--interval', type=int, default=2000, metavar='MS',
                    help='Plot refresh interval in milliseconds')

    args = ap.parse_args()

    if args.local_mode != 'tail' and args.source != 'local':
        ap.error('--replay and --full only apply to --source local')

    # ── Source ────────────────────────────────────────────────────────────────
    if args.source == 'local':
        from_start = args.local_mode in ('replay', 'full')
        source: LogSource = LocalFileTailSource(args.path, from_start=from_start)
    elif args.source == 'ssh-subprocess':
        if not args.host:
            ap.error('--host is required for ssh-subprocess')
        source = SSHSubprocessSource(args.host, args.path)
    else:
        if not args.host:
            ap.error('--host is required for paramiko')
        source = SSHParamikoSource(args.host, args.path, username=args.user, port=args.port)

    # ── Store & ingestion thread ──────────────────────────────────────────────
    window = None if args.local_mode == 'full' else args.window
    store  = DataStore(window_minutes=window)
    stop   = threading.Event()
    threading.Thread(
        target=_ingest, args=(source, store, stop),
        daemon=True, name='ingest',
    ).start()

    # ── Panels ────────────────────────────────────────────────────────────────
    panels: list[Panel] = [
        GuideOffsetsPanel(),
        FocusPanel(),
        StarQualityPanel(),
        CameraCountsPanel(),
        TelescopePanel(),
    ]

    # ── Run ───────────────────────────────────────────────────────────────────
    dashboard: Dashboard = MatplotlibDashboard(interval_ms=args.interval)
    try:
        dashboard.run(store, panels)
    finally:
        stop.set()


if __name__ == '__main__':
    main()
