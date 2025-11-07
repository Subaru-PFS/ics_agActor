import enum
import logging
import threading
import time

import numpy as np

from agActor import autoguide
from agActor.utils import actorCalls
from agActor.utils import data as data_utils
from agActor.utils import focus as _focus
from agActor.utils.actorCalls import sendAlert, send_guide_offsets
from agActor.utils.data import GuideOffsetFlag, get_guide_objects
from agActor.utils.telescope_center import telCenter as tel_center


class ag:
    class Mode(enum.IntFlag):
        # flags (Fs), inputs (Is), and states (Ss)
        OFF = 0  # [--S] idle
        ON = 1  # [FIS] start/resume autoguide
        ONCE = 2  # [FIS] autoguide once
        REF_DB = 8  # [FIS] initialize only, guide objects from opdb
        STOP = 32  # [-I-] stop autoguide
        AUTO_DB = REF_DB | ON  # [-IS] auto-start, guide objects from opdb

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    MAGNITUDE = 20.0
    DRY_RUN = False
    MAX_ELLIPTICITY = 0.6
    MAX_SIZE = 20.0  # pix
    MIN_SIZE = 0.92  # pix
    MAX_RESIDUAL = 0.2  # mm
    MAX_CORRECTION = 10  # arcsec
    EXPOSURE_DELAY = 100  # ms
    TEC_OFF = False

    class Params:
        __slots__ = (
            "mode",
            "design",
            "visit_id",
            "visit0",
            "exposure_time",
            "cadence",
            "center",
            "options",
        )

        _OPTIONS = (
            "magnitude",
            "dry_run",
            "max_ellipticity",
            "max_size",
            "min_size",
            "max_residual",
            "max_correction",
            "exposure_delay",
            "tec_off",
        )

        def __init__(self, **kwargs):
            for key in self.__slots__[:-1]:
                setattr(self, key, None)
            setattr(self, self.__slots__[-1], {})
            self.set(**kwargs)

        def set(self, **kwargs):
            for key, value in kwargs.items():
                if key in self._OPTIONS:
                    getattr(self, self.__slots__[-1])[key] = value
                else:
                    setattr(self, key, value)

        def get(self):
            return tuple(getattr(self, key) for key in self.__slots__)

    def __init__(self, actor, name, logLevel=logging.DEBUG):
        self.actor = actor
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logLevel)
        self.thread = None

    def __del__(self):
        self.logger.info("ag.__del__:")

    def start(self, cmd=None):
        self.logger.info("starting ag controller...")
        self.thread = AgThread(actor=self.actor, logger=self.logger)
        self.thread.start()

    def stop(self, cmd=None):
        self.logger.info("stopping ag controller...")
        if self.thread is not None:
            self.thread.stop()
            self.thread.join()
            self.thread = None

    def get_mode(self, cmd=None):
        mode, *_ = self.thread.get_params()
        self.logger.info(f"get_mode: {mode}")
        return mode

    def start_autoguide(
        self,
        cmd=None,
        design=None,
        visit_id=None,
        visit0=None,
        exposure_time=EXPOSURE_TIME,
        cadence=CADENCE,
        center=None,
        **kwargs,
    ):
        mode = ag.Mode.AUTO_DB

        self.logger.info(
            f"start_autoguide: {mode=},{design=},{visit_id=},{exposure_time=},{cadence=},{center=}"
        )

        self.thread.set_params(
            mode=mode,
            design=design,
            visit_id=visit_id,
            visit0=visit0,
            exposure_time=exposure_time,
            cadence=cadence,
            center=center,
            options={},
            **kwargs,
        )

    def restart_autoguide(self, cmd=None):
        mode = ag.Mode.ON
        self.logger.info(f"restart_autoguide: {mode=}")
        self.thread.set_params(mode=mode)

    def initialize_autoguide(
        self,
        cmd=None,
        design=None,
        visit_id=None,
        exposure_time=EXPOSURE_TIME,
        cadence=CADENCE,
        center=None,
        **kwargs,
    ):
        mode = ag.Mode.REF_DB

        self.logger.info(
            f"initialize_autoguide: {mode=},{design=},{visit_id=},{exposure_time=},{cadence=},{center=}"
        )

        self.thread.set_params(
            mode=mode,
            design=design,
            visit_id=visit_id,
            exposure_time=exposure_time,
            cadence=cadence,
            center=center,
            options={},
            **kwargs,
        )

    def stop_autoguide(self, cmd=None):
        self.logger.info("stop_autoguide:")
        self.thread.set_params(mode=ag.Mode.STOP)

    def reconfigure_autoguide(self, cmd=None, **kwargs):
        self.logger.info("reconfigure_autoguide:")
        self.thread.set_params(**kwargs)


class AgThread(threading.Thread):
    def __init__(self, actor=None, logger=None):
        super().__init__()

        self.actor = actor
        self.logger = logger
        self.input_params = {}
        self.params = ag.Params(
            mode=ag.Mode.OFF, exposure_time=ag.EXPOSURE_TIME, cadence=ag.CADENCE
        )
        self.logger.info("AgThread.__init__: {}".format(self.params.get()))
        self.lock = threading.Lock()
        self.__abort = threading.Event()
        self.__stop = threading.Event()

        self.with_opdb_agc_guide_offset = actor.actorConfig.get(
            "agc_guide_offset", False
        )
        self.with_opdb_agc_match = actor.actorConfig.get("agc_match", False)
        self.with_agcc_timestamp = actor.actorConfig.get("agcc_timestamp", False)
        tel_status = [
            x.strip()
            for x in actor.actorConfig.get("tel_status", ("agc_exposure",)).split(",")
        ]
        self.with_gen2_status = "gen2" in tel_status
        self.with_mlp1_status = "mlp1" in tel_status
        self.with_opdb_tel_status = "tel_status" in tel_status

    def __del__(self):
        self.logger.info("AgThread.__del__:")

    def stop(self):
        self.logger.info("AgThread.stop:")
        self.__stop.set()
        self.__abort.set()

    def _get_params(self):
        with self.lock:
            self.__abort.clear()
            self.params.set(**self.input_params)
            self.input_params.clear()
            p = self.params.get()
            return p

    def _set_params(self, **kwargs):
        self.logger.info(f"AgThread._set_params: {kwargs}")
        with self.lock:
            self.params.set(**kwargs)

    def get_params(self):
        with self.lock:
            p = self.params.get()
            self.logger.info(f"AgThread.get_params: {p}")
            return p

    def set_params(self, **kwargs):
        with self.lock:
            self.logger.info(f"AgThread.set_params: {kwargs}")
            self.input_params.update(**kwargs)
            if "mode" in kwargs:
                self.__abort.set()

    def run(self):
        cmd = self.actor.bcast

        # wait for tron to start accepting messages from this actor (~0.1 s needed)
        time.sleep(0.2)
        cmd.inform("detectionState=0")
        cmd.inform("guideReady=0")

        # Send a message to clear any previous alerts about the control loop.
        sendAlert(
            actor=self.actor,
            alert_id="AG.CONTROL_LOOP",
            alert_name="Autoguide Control Loop Started",
            alert_description="The AG control loop has been started successfully.",
            alert_severity="ok",
        )

        while True:
            if self.__stop.is_set():
                self.__stop.clear()
                self.logger.info("AgThread.run: stop has been set, setting mode to OFF")
                self._set_params(mode=ag.Mode.OFF)

            start = time.time()
            mode, design, visit_id, visit0, exposure_time, cadence, center, options = (
                self._get_params()
            )

            design_id, design_path = design if design is not None else (None, None)
            dither, offset = None, None

            guide_catalog = None

            # Make sure we have a design_id and visit_id.
            if design_id is None:
                self.logger.info("AgThread.run: no design_id, setting mode to STOP")
                self._set_params(mode=ag.Mode.STOP)
                mode = ag.Mode.STOP

            try:
                if mode & (ag.Mode.ON | ag.Mode.ONCE):
                    if guide_catalog is None:
                        self.logger.info(
                            f"Loading guide objects from database for {design_id=} {visit0=}"
                        )
                        guide_catalog = get_guide_objects(
                            design_id=design_id, visit0=visit0, is_guide=True
                        )

                    # Do the actual AG exposure.
                    exposure_delay = options.get("exposure_delay", ag.EXPOSURE_DELAY)
                    tec_off = options.get("tec_off", ag.TEC_OFF)

                    cmd.inform(f"exposureTime={exposure_time}")

                    cmd_str = f"expose object exptime={exposure_time / 1000} centroid=1"
                    if visit_id is not None:
                        cmd_str += f" visit={visit_id}"
                    if exposure_delay > 0:
                        cmd_str += f" threadDelay={exposure_delay}"
                    if tec_off:
                        cmd_str += " tecOFF"

                    self.logger.info(f"Taking AG exposure with: {cmd_str=}")
                    agcc_exposure_result = self.actor.queueCommand(
                        actor="agcc",
                        cmdStr=cmd_str,
                        timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 15),
                    )
                    time.sleep((exposure_time + 7 * exposure_delay) / 1000 / 2)

                    kwargs = {}
                    telescope_state = None
                    if self.with_mlp1_status:
                        telescope_state = self.actor.mlp1.telescopeState
                        self.logger.info(
                            f"AgThread.run: telescopeState={telescope_state}"
                        )
                        kwargs["inr"] = telescope_state["rotator_real_angle"]

                    # update gen2 status values.
                    if self.with_gen2_status or self.with_opdb_tel_status:
                        self.logger.info("AgThread.run: getting gen2 status")
                        if self.with_gen2_status:
                            try:
                                tel_status = actorCalls.updateTelStatus(
                                    self.actor, self.logger, visit_id
                                )
                            except Exception as e:
                                # Raising a RuntimeError as this call is usually not fatal.
                                raise RuntimeError(
                                    f"AgThread.updateTelStatus error: {e}"
                                )

                            self.logger.info(f"AgThread.run: {tel_status=}")
                            kwargs["tel_status"] = tel_status
                            _tel_center = tel_center(
                                actor=self.actor,
                                center=center,
                                design=design,
                                tel_status=tel_status,
                            )
                            if all(x is None for x in (center, design)):
                                # dithered center and guide offset correction (insrot only)
                                center, offset = _tel_center.dither
                                self.logger.info(f"AgThread.run: {center=}")
                            else:
                                # dithering and guide offset correction
                                offset = _tel_center.offset
                            self.logger.info(f"AgThread.run: {offset=}")

                        if self.with_opdb_tel_status:
                            status_update = self.actor.gen2.statusUpdate
                            status_id = (
                                status_update["visit"],
                                status_update["sequenceNum"],
                            )
                            self.logger.info(f"AgThread.run: status_id={status_id}")
                            kwargs["status_id"] = status_id
                    # wait for the exposure to complete.
                    agcc_exposure_result.get()

                    data_time = self.actor.agcc.dataTime
                    self.logger.info(f"AgThread.run: dataTime={data_time}")
                    taken_at = (
                        data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2
                    )
                    self.logger.info(f"AgThread.run: taken_at={taken_at}")
                    if self.with_agcc_timestamp:
                        # unix timestamp, not timezone-aware datetime
                        kwargs["taken_at"] = taken_at
                    if self.with_mlp1_status:
                        # possibly override timestamp from agcc
                        taken_at = self.actor.mlp1.setUnixDay(
                            telescope_state["az_el_detect_time"], taken_at
                        )
                        kwargs["taken_at"] = taken_at

                    if center is not None:
                        kwargs["center"] = center
                    if offset is not None:
                        kwargs["offset"] = offset

                    max_correction = options.get("max_correction", ag.MAX_CORRECTION)

                    if "max_ellipticity" in options:
                        kwargs["max_ellipticity"] = options.get("max_ellipticity")
                    if "max_size" in options:
                        kwargs["max_size"] = options.get("max_size")
                    if "min_size" in options:
                        kwargs["min_size"] = options.get("min_size")
                    if "max_residual" in options:
                        kwargs["max_residual"] = options.get("max_residual")

                    # Compute guide errors for exposure.
                    cmd.inform("detectionState=1")
                    frame_id = self.actor.agcc.frameId
                    self.logger.info(
                        f"AgThread.run: autoguide.autoguide for {frame_id=}"
                    )
                    guide_offsets = autoguide.get_exposure_offsets(
                        frame_id=frame_id, guide_catalog=guide_catalog, **kwargs
                    )

                    # Extract values from the AutoguideResult dataclass
                    ra = guide_offsets.ra
                    dec = guide_offsets.dec
                    inst_pa = guide_offsets.inst_pa
                    dra = guide_offsets.ra_offset
                    ddec = guide_offsets.dec_offset
                    dinr = guide_offsets.inr_offset
                    dscale = guide_offsets.scale_offset
                    dalt = guide_offsets.dalt
                    daz = guide_offsets.daz
                    cmd.inform(
                        f'text="{ra=},{dec=},{inst_pa=},{dra=},{ddec=},{dinr=},{dscale=},{dalt=},{daz=}"'
                    )

                    filenames = guide_offsets.save_numpy_files()
                    cmd.inform(
                        'data={},{},{},"{}","{}","{}"'.format(
                            ra, dec, inst_pa, *filenames
                        )
                    )
                    cmd.inform("detectionState=0")

                    dx = guide_offsets.dx
                    dy = guide_offsets.dy
                    size = guide_offsets.size
                    peak = guide_offsets.peak
                    flux = guide_offsets.flux
                    self.logger.info(
                        f"AgThread.run: Sending mlp1 command {dx=},{dy=},{size=},{peak=},{flux=}"
                    )

                    offset_in_range = (
                        abs(dra) < max_correction and abs(ddec) < max_correction
                    )

                    if offset_in_range:
                        offset_flags = GuideOffsetFlag.OK
                        guide_status = "OK"
                    else:
                        offset_flags = GuideOffsetFlag.INVALID_OFFSET
                        guide_status = f"INVALID_OFFSET"

                    if offset_flags == GuideOffsetFlag.OK:
                        # send corrections to mlp1 and gen2 (or iic).
                        dry_run = options.get("dry_run", ag.DRY_RUN)
                        send_guide_offsets(
                            actor=self.actor,
                            taken_at=taken_at,
                            daz=daz,
                            dalt=dalt,
                            dx=dx,
                            dy=dy,
                            size=size,
                            peak=peak,
                            flux=flux,
                            dry_run=dry_run,
                            logger=self.actor.logger,
                        )
                    else:
                        cmd.inform(
                            f'text="Calculated offset not in allowed range, skipping: {dra=} {ddec=} {max_correction=}"'
                        )
                        sendAlert(
                            actor=self.actor,
                            alert_id="AG.OFFSET_OUT_OF_RANGE",
                            alert_name="Autoguide Offset Out of Range",
                            alert_description="The calculated autoguide offset is out of the allowed range, no corrections have been sent to the telescope.",
                            alert_detail=f"Calculated offsets: {frame_id=} {visit_id=} {dra=}, {ddec=}, {max_correction=}",
                            alert_severity="warning",
                        )

                    # always compute focus offset and tilt.
                    self.logger.info(
                        f"AgThread.run: focus._focus for frame_id={frame_id}"
                    )

                    # TODO (wtg - 2025-10-16): deal with max_ellipticity, max_size, min_size.
                    dz, dzs = _focus._focus(
                        detected_objects=guide_offsets.detected_objects
                    )

                    # send corrections to gen2 (or iic).
                    if dalt is None:
                        dalt = np.nan
                    if daz is None:
                        daz = np.nan
                    cmd.inform(
                        "guideErrors={},{},{},{},{},{},{},{},{}".format(
                            frame_id,
                            dra,
                            ddec,
                            dinr,
                            daz,
                            dalt,
                            dz,
                            dscale,
                            guide_status,
                        )
                    )
                    cmd.inform(
                        "focusErrors={},{},{},{},{},{},{}".format(frame_id, *dzs)
                    )

                    if self.with_opdb_agc_guide_offset:
                        self.logger.info(
                            f"AgThread.run: write_agc_guide_offset for {frame_id=}"
                        )
                        data_utils.write_agc_guide_offset(
                            frame_id=frame_id,
                            ra=ra,
                            dec=dec,
                            pa=inst_pa,
                            delta_ra=dra,
                            delta_dec=ddec,
                            delta_insrot=dinr,
                            delta_scale=dscale,
                            delta_az=daz,
                            delta_el=dalt,
                            delta_z=dz,
                            delta_zs=dzs,
                            offset_flags=offset_flags,
                        )
                    if self.with_opdb_agc_match:
                        self.logger.info(
                            f"AgThread.run: write_agc_match for {frame_id=}"
                        )
                        data_utils.write_agc_match(
                            design_id=design_id,
                            frame_id=frame_id,
                            guide_objects=guide_offsets.guide_objects,
                            detected_objects=guide_offsets.detected_objects,
                            identified_objects=guide_offsets.identified_objects,
                        )

                if mode & ag.Mode.ONCE:
                    self.logger.info("AgThread.run: ONCE")
                    self._set_params(mode=ag.Mode.OFF)
                if mode == ag.Mode.STOP:
                    self.logger.info("AgThread.run: STOP")
                    cmd.inform("guideReady=0")
                    self._set_params(mode=ag.Mode.OFF)
            except RuntimeError as e:
                self.logger.error(f"AgThread.run: RuntimeError: {e}")
                sendAlert(
                    actor=self.actor,
                    alert_id="AG.CONTROL_LOOP.RUNTIME_ERROR",
                    alert_name="Autoguide Control-loop Runtime Error",
                    alert_description="Non-fatal error occurred, continuing to next iteration (see Details).",
                    alert_detail=str(e),
                    alert_severity="warning",
                )
                self.logger.warning(
                    "AgThread.run: Going to next iteration because of non-fatal error"
                )
            except Exception as e:
                self.logger.error(f"AgThread.run error: {e}")
                self.logger.error("AgThread.run: stopping run loop due to error")
                sendAlert(
                    actor=self.actor,
                    alert_id="AG.CONTROL_LOOP",
                    alert_name="Autoguide Fatal Error",
                    alert_description="A fatal error occurred, autoguiding has been stopped.",
                    alert_detail=str(e),
                    alert_severity="critical",
                )
                self.stop()

            end = time.time()
            timeout = (
                max(0, cadence / 1000 - (end - start)) if mode == ag.Mode.ON else 0.5
            )
            self.__abort.wait(timeout)

        cmd.inform("guideReady=0")
