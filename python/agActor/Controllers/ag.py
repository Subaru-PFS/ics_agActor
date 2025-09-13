import enum
import logging
import threading
import time

import numpy as np

from agActor import autoguide
from agActor.catalog import pfs_design
from agActor.utils import actorCalls
from agActor.utils import data as data_utils
from agActor.utils import focus as _focus
from agActor.utils.data import GuideOffsetFlag
from agActor.utils.telescope_center import telCenter as tel_center


class ag:
    class Mode(enum.IntFlag):
        # flags (Fs), inputs (Is), and states (Ss)
        OFF = 0  # [--S] idle
        ON = 1  # [FIS] start/resume autoguide
        ONCE = 2  # [FIS] autoguide once
        REF_SKY = 4  # [FIS] initialize only, guide objects from exposure
        REF_DB = 8  # [FIS] initialize only, guide objects from opdb
        REF_OTF = 16  # [FIS] initialize only, guide objects from catalog on the fly
        STOP = 32  # [-I-] stop autoguide
        AUTO_SKY = REF_SKY | ON  # [-IS] auto-start, guide objects from first exposure
        AUTO_DB = REF_DB | ON  # [-IS] auto-start, guide objects from opdb
        AUTO_OTF = (
            REF_OTF | ON
        )  # [-IS] auto-start, guide objects from catalog on the fly
        # AUTO_ONCE_SKY = REF_SKY | ONCE  # [-IS] initialize and autoguide once, guide objects from exposure
        AUTO_ONCE_DB = (
            REF_DB | ONCE
        )  # [-IS] initialize and autoguide once, guide objects from opdb
        #       (field acquisition with autoguider)
        AUTO_ONCE_OTF = (
            REF_OTF | ONCE
        )  # [-IS] initialize and autoguide once, guide objects from catalog on the fly
        #       (on-the-fly field acquisition with autoguider)

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    MAGNITUDE = 20.0
    DRY_RUN = False
    FIT_DINR = True
    FIT_DSCALE = False
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
            "exposure_time",
            "cadence",
            "center",
            "options",
        )

        _OPTIONS = (
            "magnitude",
            "dry_run",
            "fit_dinr",
            "fit_dscale",
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
        from_sky=None,
        exposure_time=EXPOSURE_TIME,
        cadence=CADENCE,
        center=None,
        **kwargs,
    ):
        mode = (
            ag.Mode.AUTO_SKY
            if from_sky
            else ag.Mode.AUTO_DB
            if design is not None
            else ag.Mode.AUTO_OTF
        )
        self.logger.info(
            f"start_autoguide: {mode=},{design=},{visit_id=},{exposure_time=},{cadence=},{center=}"
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

    def restart_autoguide(self, cmd=None):
        mode = ag.Mode.ON
        self.logger.info(f"restart_autoguide: {mode=}")
        self.thread.set_params(mode=mode)

    def initialize_autoguide(
        self,
        cmd=None,
        design=None,
        visit_id=None,
        from_sky=None,
        exposure_time=EXPOSURE_TIME,
        cadence=CADENCE,
        center=None,
        **kwargs,
    ):
        mode = (
            ag.Mode.REF_SKY
            if from_sky
            else ag.Mode.REF_DB
            if design is not None
            else ag.Mode.REF_OTF
        )
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

    def acquire_field(
        self,
        cmd=None,
        design=None,
        visit_id=None,
        exposure_time=EXPOSURE_TIME,
        center=None,
        **kwargs,
    ):
        mode = ag.Mode.AUTO_ONCE_DB if design is not None else ag.Mode.AUTO_ONCE_OTF
        self.logger.info(
            f"acquire_field: {mode=},{design=},{visit_id=},{exposure_time=},{center=}"
        )
        self.thread.set_params(
            mode=mode,
            design=design,
            visit_id=visit_id,
            exposure_time=exposure_time,
            center=center,
            options={},
            **kwargs,
        )


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

        while True:
            if self.__stop.is_set():
                self.__stop.clear()
                self.logger.info("AgThread.run: stop has been set, setting mode to OFF")
                self._set_params(mode=ag.Mode.OFF)

            start = time.time()
            mode, design, visit_id, exposure_time, cadence, center, options = (
                self._get_params()
            )
            design_id, design_path = design if design is not None else (None, None)
            dither, offset = None, None
            try:
                if mode & ag.Mode.REF_OTF and not mode & (ag.Mode.ON | ag.Mode.ONCE):
                    self.logger.info("AgThread.run: REF_OTF")
                    if self.with_gen2_status:
                        self.logger.info("AgThread.run: gen2 status")
                        kwargs = {}
                        if self.with_mlp1_status:
                            self.logger.info("AgThread.run: mlp1 status")
                            telescope_state = self.actor.mlp1.telescopeState
                            self.logger.info(
                                f"AgThread.run: telescopeState={telescope_state}"
                            )
                            kwargs["inr"] = telescope_state["rotator_real_angle"]
                        # update gen2 status values
                        self.logger.info("AgThread.run: update gen2 status")
                        tel_status = actorCalls.updateTelStatus(
                            self.actor, self.logger, visit_id
                        )
                        self.logger.info(f"AgThread.run: tel_status={tel_status}")
                        kwargs["tel_status"] = tel_status
                        _tel_center = tel_center(
                            actor=self.actor,
                            center=center,
                            design=design,
                            tel_status=tel_status,
                        )
                        if center is None:
                            center, offset = (
                                _tel_center.dither
                            )  # dithered center and guide offset correction (insrot only)
                            self.logger.info(f"AgThread.run: center={center}")
                        else:
                            offset = (
                                _tel_center.offset
                            )  # dithering and guide offset correction
                        self.logger.info(f"AgThread.run: offset={offset}")
                        if self.with_mlp1_status:
                            self.logger.info("AgThread.run: mlp1 status")
                            taken_at = self.actor.mlp1.setUnixDay(
                                telescope_state["az_el_detect_time"],
                                tel_status[8].timestamp(),
                            )
                            kwargs["taken_at"] = taken_at
                        if center is not None:
                            kwargs["center"] = center
                        if offset is not None:
                            kwargs["offset"] = offset
                        if "magnitude" in options:
                            kwargs["magnitude"] = options.get("magnitude")

                        self.logger.info(
                            "AgThread.run: REF_OTF set_design and set_design_ac"
                        )
                        autoguide.set_design(logger=self.logger, **kwargs)
                        autoguide.set_design_agc(logger=self.logger, **kwargs)
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)
                if mode & ag.Mode.REF_DB:
                    self.logger.info("AgThread.run: Setting design for REF_DB")
                    autoguide.set_design(design=design, logger=self.logger)
                    kwargs = {}
                    if "magnitude" in options:
                        kwargs["magnitude"] = options.get("magnitude")

                    self.logger.info(
                        "AgThread.run: REF_DB set_design and set_design_ac"
                    )
                    autoguide.set_design_agc(
                        logger=self.logger, **kwargs
                    )  # obstime=<current time>
                    mode &= ~ag.Mode.REF_DB
                    self._set_params(mode=mode)
                if mode & (ag.Mode.ON | ag.Mode.ONCE | ag.Mode.REF_SKY):
                    self.logger.info("AgThread.run: Setting design for ON/ONCE/REF_SKY")
                    exposure_delay = options.get("exposure_delay", ag.EXPOSURE_DELAY)
                    tec_off = options.get("tec_off", ag.TEC_OFF)
                    cmd.inform("exposureTime={}".format(exposure_time))
                    # start an exposure
                    cmdStr = "expose object exptime={} centroid=1".format(
                        exposure_time / 1000
                    )
                    if visit_id is not None:
                        cmdStr += " visit={}".format(visit_id)
                    if exposure_delay > 0:
                        cmdStr += " threadDelay={}".format(exposure_delay)
                    if tec_off:
                        cmdStr += " tecOFF"

                    self.logger.info(
                        "Taking AG exposure with: cmdStr={}".format(cmdStr)
                    )
                    agcc_exposure_result = self.actor.queueCommand(
                        actor="agcc",
                        cmdStr=cmdStr,
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
                    if self.with_gen2_status or self.with_opdb_tel_status:
                        # update gen2 status values
                        self.logger.info("AgThread.run: getting gen2 status")
                        if self.with_gen2_status:
                            tel_status = actorCalls.updateTelStatus(
                                self.actor, self.logger, visit_id
                            )
                            self.logger.info(f"AgThread.run: tel_status={tel_status}")
                            kwargs["tel_status"] = tel_status
                            _tel_center = tel_center(
                                actor=self.actor,
                                center=center,
                                design=design,
                                tel_status=tel_status,
                            )
                            if all(x is None for x in (center, design)):
                                center, offset = (
                                    _tel_center.dither
                                )  # dithered center and guide offset correction (insrot only)
                                self.logger.info(f"AgThread.run: center={center}")
                            else:
                                offset = (
                                    _tel_center.offset
                                )  # dithering and guide offset correction
                            self.logger.info(f"AgThread.run: offset={offset}")
                        if self.with_opdb_tel_status:
                            status_update = self.actor.gen2.statusUpdate
                            status_id = (
                                status_update["visit"],
                                status_update["sequenceNum"],
                            )
                            self.logger.info(f"AgThread.run: status_id={status_id}")
                            kwargs["status_id"] = status_id
                    # wait for an exposure to complete
                    agcc_exposure_result.get()
                    frame_id = self.actor.agcc.frameId
                    self.logger.info(f"AgThread.run: frameId={frame_id}")
                    data_time = self.actor.agcc.dataTime
                    self.logger.info(f"AgThread.run: dataTime={data_time}")
                    taken_at = (
                        data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2
                    )
                    self.logger.info(f"AgThread.run: taken_at={taken_at}")
                    if self.with_agcc_timestamp:
                        kwargs["taken_at"] = (
                            taken_at  # unix timestamp, not timezone-aware datetime
                        )
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
                    if "magnitude" in options:
                        kwargs["magnitude"] = options.get("magnitude")

                    self.logger.info(f"AgThread.run: kwargs={kwargs}")

                    if mode & ag.Mode.REF_OTF:
                        self.logger.info(
                            "AgThread.run: REF_OTF set_design and set_design_ac"
                        )
                        autoguide.set_design(logger=self.logger, **kwargs)
                        autoguide.set_design_agc(logger=self.logger, **kwargs)
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)
                    # retrieve detected objects from opdb
                    if mode & ag.Mode.REF_SKY:
                        self.logger.info(
                            "AgThread.run: REF_SKY set_design and set_design_ac"
                        )
                        # store initial conditions
                        autoguide.set_design(
                            design=design, logger=self.logger, **kwargs
                        )  # center takes precedence over design
                        autoguide.set_design_agc(
                            frame_id=frame_id, logger=self.logger, **kwargs
                        )
                        mode &= ~ag.Mode.REF_SKY
                        self._set_params(mode=mode)
                    else:  # mode & (ag.Mode.ON | ag.Mode.ONCE)
                        self.logger.info(
                            "AgThread.run: ON/ONCE set_design and set_design_ac"
                        )
                        dry_run = options.get("dry_run", ag.DRY_RUN)
                        if "fit_dinr" in options:
                            kwargs["fit_dinr"] = options.get("fit_dinr")
                        if "fit_dscale" in options:
                            kwargs["fit_dscale"] = options.get("fit_dscale")
                        if "max_ellipticity" in options:
                            kwargs["max_ellipticity"] = options.get("max_ellipticity")
                        if "max_size" in options:
                            kwargs["max_size"] = options.get("max_size")
                        if "min_size" in options:
                            kwargs["min_size"] = options.get("min_size")
                        if "max_residual" in options:
                            kwargs["max_residual"] = options.get("max_residual")
                        max_correction = options.get(
                            "max_correction", ag.MAX_CORRECTION
                        )

                        cmd.inform("detectionState=1")
                        # compute guide errors
                        self.logger.info(
                            f"AgThread.run: autoguide.autoguide for frame_id={frame_id}"
                        )
                        guide_offsets = autoguide.autoguide(
                            frame_id=frame_id, logger=self.logger, **kwargs
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
                            'text="ra={},dec={},inst_pa={},dra={},ddec={},dinr={},dscale={},dalt={},daz={}"'.format(
                                ra, dec, inst_pa, dra, ddec, dinr, dscale, dalt, daz
                            )
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
                            # send corrections to mlp1 and gen2 (or iic)
                            mlp1_result = self.actor.queueCommand(
                                actor="mlp1",
                                # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
                                cmdStr="guide azel={},{} ready={} time={} delay=0 xy={},{} size={} intensity={} flux={}".format(
                                    -daz,
                                    -dalt,
                                    int(not dry_run),
                                    taken_at,
                                    dx * 1e3,
                                    -dy * 1e3,
                                    size,
                                    peak,
                                    flux,
                                ),
                                timeLim=5,
                            )
                            mlp1_result.get()
                        else:
                            cmd.inform(
                                f'text="Calculated offset not in allowed range, skipping: {dra=} {ddec=} {max_correction=}"'
                            )

                        kwargs = {
                            key: kwargs.get(key)
                            for key in ("max_ellipticity", "max_size", "min_size")
                            if key in kwargs
                        }
                        # always compute focus offset and tilt
                        self.logger.info(
                            f"AgThread.run: focus._focus for frame_id={frame_id}"
                        )
                        dz, dzs = _focus._focus(
                            detected_objects=guide_offsets.detected_objects,
                            logger=self.logger,
                            **kwargs,
                        )
                        # send corrections to gen2 (or iic)
                        if dalt is None:
                            dalt = np.nan
                        if daz is None:
                            daz = np.nan
                        cmd.inform(
                            "guideErrors={},{},{},{},{},{},{},{},{}".format(
                                frame_id, dra, ddec, dinr, daz, dalt, dz, dscale, guide_status
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
                                f"AgThread.run: write_agc_match for frame_id={frame_id}"
                            )
                            data_utils.write_agc_match(
                                design_id=(
                                    design_id
                                    if design_id is not None
                                    else (
                                        pfs_design.pfsDesign.to_design_id(design_path)
                                        if design_path is not None
                                        else 0
                                    )
                                ),
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
            except Exception as e:
                self.logger.error(f"AgThread.run error: {e}")
                self.logger.error("AgThread.run: stopping run loop due to error")
                self.stop()

            end = time.time()
            timeout = (
                max(0, cadence / 1000 - (end - start)) if mode == ag.Mode.ON else 0.5
            )
            self.__abort.wait(timeout)

        cmd.inform("guideReady=0")
