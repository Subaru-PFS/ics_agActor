import enum
import logging
import threading
import time

from opscore.utility.qstr import qstr

from agActor.exposure import run_exposure_pipeline
from agActor.utils.actorCalls import sendAlert
from agActor.utils.data import get_guide_objects


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
    FIT_DINR = True
    FIT_DSCALE = False
    MAX_ELLIPTICITY = 2.0
    MAX_SIZE = 1.0e12  # pix
    MIN_SIZE = -1.0e0  # pix
    MAX_RESIDUAL = 0.2  # mm
    MAX_CORRECTION = 10  # arcsec
    EXPOSURE_DELAY = 100  # ms
    TEC_OFF = False
    FILTER_BAD_SHAPE = True # By default we don't want to filter bad shape objects.

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
            "fit_dinr",
            "fit_dscale",
            "max_ellipticity",
            "max_size",
            "min_size",
            "max_residual",
            "max_correction",
            "exposure_delay",
            "tec_off",
            "filter_bad_shape",
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
        from_sky=None,
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
        from_sky=None,
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
            if p[0] != ag.Mode.OFF:
                self.logger.info(f"AgThread._get_params: {p}")
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
            logger=self.actor.logger,
        )

        while True:
            if self.__stop.is_set():
                self.__stop.clear()
                self.logger.info("AgThread.run: stop has been set, setting mode to OFF")
                self._set_params(mode=ag.Mode.OFF)

            try:
                start = time.time()
                mode, design, visit_id, visit0, exposure_time, cadence, center, options = (
                    self._get_params()
                )

                design_id, design_path = design if design is not None else (None, None)

                guide_catalog = None
            except Exception as e:
                self.logger.error(f"AgThread.run error during parameter fetch: {e}")
                self.logger.error("AgThread.run: stopping run loop due to error")
                sendAlert(
                    actor=self.actor,
                    alert_id="AG.CONTROL_LOOP",
                    alert_name="Autoguide Fatal Error",
                    alert_description="A fatal error occurred while fetching parameters, autoguiding has been stopped.",
                    alert_detail=str(e),
                    alert_severity="critical",
                    logger=self.actor.logger,
                )
                break

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
                    max_correction = options.get("max_correction", ag.MAX_CORRECTION)
                    max_ellipticity = options.get("max_ellipticity", ag.MAX_ELLIPTICITY)
                    max_size = options.get("max_size", ag.MAX_SIZE)
                    min_size = options.get("min_size", ag.MIN_SIZE)
                    dry_run = options.get("dry_run", ag.DRY_RUN)

                    pipeline_kwargs = {}
                    if "max_residual" in options:
                        pipeline_kwargs["max_residual"] = options["max_residual"]
                    if "filter_bad_shape" in options:
                        pipeline_kwargs["filter_bad_shape"] = options["filter_bad_shape"]

                    run_exposure_pipeline(
                        actor=self.actor,
                        cmd=cmd,
                        cfg=self.actor.ag_config,
                        design_id=design_id,
                        design_path=design_path,
                        design=design,
                        visit_id=visit_id,
                        visit0=visit0,
                        exposure_time=exposure_time,
                        exposure_delay=exposure_delay,
                        tec_off=tec_off,
                        center=center,
                        guide_catalog=guide_catalog,
                        send_offsets=True,
                        dry_run=dry_run,
                        max_correction=max_correction,
                        max_ellipticity=max_ellipticity,
                        max_size=max_size,
                        min_size=min_size,
                        **pipeline_kwargs,
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
                    alert_detail=qstr(e),
                    alert_severity="warning",
                    logger=self.actor.logger,
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
                    alert_detail=qstr(e),
                    alert_severity="critical",
                    logger=self.actor.logger,
                )
                self.stop()

            end = time.time()
            elapsed = end - start
            self.logger.info(f"AgThread.run: iteration took {elapsed:.3f} s")
            timeout = (max(0, cadence / 1000 - elapsed) if mode & ag.Mode.ON else 0.5)
            self.logger.info(f"AgThread.run: Control loop delay {timeout=:.3f} {self.__abort.is_set()=}")
            self.__abort.wait(timeout)

        cmd.inform("guideReady=0")
