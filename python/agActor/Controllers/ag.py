import enum
import logging
import threading
import time
import numpy
from agActor import autoguide


class ag:

    class Mode(enum.IntFlag):

        # flags (Fs), inputs (Is), and states (Ss)
        OFF = 0                         # [--S] idle
        ON = 1                          # [FIS] start/resume autoguide
        ONCE = 2                        # [FIS] autoguide once
        REF = 4                         # [FIS] initialize only, guide objects from exposure
        DB = 8                          # [F--] guide objects from opdb
        STOP = 16                       # [-I-] stop autoguide
        REF_DB = REF | DB               # [-IS] initialize only, guide objects from opdb
        AUTO = REF | ON                 # [-IS] auto-start, guide objects from first exposure
        AUTO_DB = REF | DB | ON         # [-IS] auto-start, guide objects from opdb
        AUTO_ONCE = REF | ONCE          # [-IS] initialize and autoguide once, guide objects from exposure
        AUTO_ONCE_DB = REF | DB | ONCE  # [-IS] initialize and autoguide once, guide objects from opdb
                                        #       (field acquisition with autoguider)

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    FOCUS = False

    class Params:

        def __init__(self, **kwargs):

            self.set(**kwargs)

        def set(self, mode=None, design=None, visit_id=None, exposure_time=None, cadence=None, focus=None):

            self.mode = mode
            self.design = design
            self.visit_id = visit_id
            self.exposure_time = exposure_time
            self.cadence = cadence
            self.focus = focus

        def reset(self, mode=None, design=None, visit_id=None, exposure_time=None, cadence=None, focus=None):

            if mode is not None: self.mode = mode
            if design is not None: self.design = design
            if visit_id is not None: self.visit_id = visit_id
            if exposure_time is not None: self.exposure_time = exposure_time
            if cadence is not None: self.cadence = cadence
            if focus is not None: self.focus = focus

        def clear(self):

            self.set()

        def get(self):

            return self.mode, self.design, self.visit_id, self.exposure_time, self.cadence, self.focus

    def __init__(self, actor, name, logLevel=logging.DEBUG):

        self.actor = actor
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logLevel)
        self.thread = None

    def __del__(self):

        self.logger.info('ag.__del__:')

    def start(self, cmd=None):

        self.logger.info('starting ag controller...')
        self.thread = AgThread(actor=self.actor, logger=self.logger)
        self.thread.start()

    def stop(self, cmd=None):

        self.logger.info('stopping ag controller...')
        if self.thread is not None:
            self.thread.stop()
            self.thread.join()
            self.thread = None

    def get_mode(self, cmd=None):

        mode, _, _, _, _, _ = self.thread.get_params()
        return mode

    def start_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME, cadence=CADENCE, focus=FOCUS):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO if from_sky else ag.Mode.ON if design is None else ag.Mode.AUTO_DB
        self.thread.set_params(mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, cadence=cadence, focus=focus)

    def initialize_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.REF if from_sky else ag.Mode.REF_DB
        self.thread.set_params(mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time)

    def stop_autoguide(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.STOP)

    def reconfigure_autoguide(self, cmd=None, exposure_time=None, cadence=None, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(exposure_time=exposure_time, cadence=cadence, focus=focus)

    def acquire_field(self, cmd=None, design=None, visit_id=None, exposure_time=EXPOSURE_TIME, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.AUTO_ONCE_DB, design=design, visit_id=visit_id, exposure_time=exposure_time, focus=focus)


class AgThread(threading.Thread):

    def __init__(self, actor=None, logger=None):

        super().__init__()

        self.actor = actor
        self.logger = logger
        self.input_params = ag.Params()
        self.params = ag.Params(mode=ag.Mode.OFF, exposure_time=ag.EXPOSURE_TIME, cadence=ag.CADENCE, focus=ag.FOCUS)
        self.lock = threading.Lock()
        self.__abort = threading.Event()
        self.__stop = threading.Event()

    def __del__(self):

        self.logger.info('AgThread.__del__:')

    def stop(self):

        self.__stop.set()
        self.__abort.set()

    def _get_params(self):

        with self.lock:
            self.__abort.clear()
            self.params.reset(*self.input_params.get())
            self.input_params.clear()
            return self.params.get()

    def _set_params(self, **kwargs):

        with self.lock:
            self.params.reset(**kwargs)

    def get_params(self):

        with self.lock:
            return self.params.get()

    def set_params(self, mode=None, **kwargs):

        with self.lock:
            self.input_params.reset(mode=mode, **kwargs)
            if mode is not None:
                self.__abort.set()

    def run(self):

        cmd = self.actor.bcast

        time.sleep(0.2)  # wait for tron to start accepting messages from this actor (~0.1 s needed)
        cmd.inform('detectionState=0')
        cmd.inform('guideReady=0')

        while True:

            if self.__stop.is_set():
                self.__stop.clear()
                break
            start = time.time()
            mode, design, visit_id, exposure_time, cadence, focus = self._get_params()
            design_id, design_path = design if design is not None else (None, None)
            self.logger.info('AgThread.run: mode={},design={},visit_id={},exposure_time={},cadence={},focus={}'.format(mode, design, visit_id, exposure_time, cadence, focus))
            try:
                if mode & ag.Mode.REF:
                    autoguide.set_design(design=(design_id, design_path), logger=self.logger)
                    if mode & ag.Mode.DB:
                        autoguide.set_design_agc(logger=self.logger)
                        mode &= ~(ag.Mode.REF | ag.Mode.DB)
                        self._set_params(mode=mode)
                if mode & (ag.Mode.ON | ag.Mode.ONCE | ag.Mode.REF):
                    cmd.inform('exposureTime={}'.format(exposure_time))
                    result = self.actor.sendCommand(
                        actor='agcc',
                        cmdStr='expose object pfsVisitId={} exptime={} centroid=1'.format(visit_id, exposure_time / 1000),
                        timeLim=(exposure_time // 1000 + 5)
                    )
                    telescope_state = self.actor.mlp1.telescopeState
                    self.logger.info('AgThread.run: telescopeState={}'.format(telescope_state))
                    frame_id = self.actor.agcc.frameId
                    self.actor.logger.info('AgThread.run: frameId={}'.format(frame_id))
                    data_time = self.actor.agcc.dataTime
                    self.logger.info('AgThread.run: dataTime={}'.format(data_time))
                    # retrieve detected objects from opdb
                    if mode & ag.Mode.REF:
                        # store initial conditions
                        autoguide.set_design_agc(frame_id=frame_id, logger=self.logger)
                        self._set_params(mode=mode & ~ag.Mode.REF)
                    else:  # mode & (ag.Mode.ON | ag.Mode.ONCE)
                        cmd.inform('detectionState=1')
                        # compute guide errors
                        dalt, daz, _, *values = autoguide.autoguide(frame_id=frame_id, verbose=True, logger=self.logger)
                        filename = '/dev/shm/guide_objects.npy'
                        numpy.save(filename, values[0])
                        cmd.inform('guideObjects={}'.format(filename))
                        filename = '/dev/shm/detected_objects.npy'
                        numpy.save(filename, values[1])
                        cmd.inform('detectedObjects={}'.format(filename))
                        filename = '/dev/shm/identified_objects.npy'
                        numpy.save(filename, values[2])
                        cmd.inform('identifiedObjects={}'.format(filename))
                        cmd.inform('detectionState=0')
                        dx, dy, size, peak, flux = values[3], values[4], values[5], values[6], values[7]
                        result = self.actor.sendCommand(
                            actor='mlp1',
                            # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
                            cmdStr='guide azel={},{} ready=1 time={} delay=0 xy={},{} size={} intensity={} flux={}'.format(- daz, - dalt, data_time, dx / 0.098, - dy / 0.098, size * 13 / 0.098, peak, flux),
                            timeLim=5
                        )
                        #cmd.inform('guideReady=1')
                        if focus:
                            # compute focus error
                            pass
                if mode & ag.Mode.ONCE:
                    self._set_params(mode=ag.Mode.OFF)
                if mode == ag.Mode.STOP:
                    #cmd.inform('detectionState=0')
                    cmd.inform('guideReady=0')
                    self._set_params(mode=ag.Mode.OFF)
            except Exception as e:
                self.logger.error('AgThread.run: {}'.format(e))
            end = time.time()
            timeout = max(0, cadence / 1000 - (end - start)) if mode == ag.Mode.ON else 0.5
            self.__abort.wait(timeout)

        cmd.inform('guideReady=0')
