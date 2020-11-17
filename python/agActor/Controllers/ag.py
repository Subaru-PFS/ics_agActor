import enum
import logging
import threading
import time
from agActor import autoguide


class ag:

    class Mode(enum.Enum):

        OFF = 0
        ON = 1
        INIT = INIT_SKY = 2
        AUTO = AUTO_SKY = 3
        INIT_DB = 4
        AUTO_DB = 5
        STOP = 6

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    FOCUS = False

    class Params:

        def __init__(self, **kwargs):

            self.set(**kwargs)

        def set(self, mode=None, target_id=None, exposure_time=None, cadence=None, focus=None):

            self.mode = mode
            self.target_id = target_id
            self.exposure_time = exposure_time
            self.cadence = cadence
            self.focus = focus

        def update(self, mode=None, target_id=None, exposure_time=None, cadence=None, focus=None):

            if mode is not None: self.mode = mode
            if target_id is not None: self.target_id = target_id
            if exposure_time is not None: self.exposure_time = exposure_time
            if cadence is not None: self.cadence = cadence
            if focus is not None: self.focus = focus

        def release(self):

            params = {'mode': self.mode, 'target_id': self.target_id, 'exposure_time': self.exposure_time, 'cadence': self.cadence, 'focus': self.focus}
            self.set()  # set all members to None
            return params

        def get(self):

            return self.mode, self.target_id, self.exposure_time, self.cadence, self.focus

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

        mode, _, _, _, _ = self.thread.params.get()
        return mode

    def start_autoguide(self, cmd=None, target_id=None, from_sky=None, exposure_time=EXPOSURE_TIME, cadence=CADENCE, focus=FOCUS):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO if from_sky else ag.Mode.ON if target_id is None else ag.Mode.AUTO_DB
        self.thread.set_params(mode=mode, target_id=target_id, exposure_time=exposure_time, cadence=cadence, focus=focus)

    def initialize_autoguide(self, cmd=None, target_id=None, from_sky=None, exposure_time=EXPOSURE_TIME):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.INIT if from_sky else ag.Mode.INIT_DB
        self.thread.set_params(mode=mode, target_id=target_id, exposure_time=exposure_time)

    def stop_autoguide(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.STOP)

    def reconfigure_autoguide(self, cmd=None, exposure_time=None, cadence=None, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(exposure_time=exposure_time, cadence=cadence, focus=focus)


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
            self.params.update(**self.input_params.release())
            return self.params.get()

    def _set_params(self, **kwargs):

        with self.lock:
            self.params.update(**kwargs)

    def get_params(self):

        with self.lock:
            return self.params.get()

    def set_params(self, mode=None, **kwargs):

        with self.lock:
            self.input_params.update(mode=mode, **kwargs)
            if mode is not None:
                self.__abort.set()

    def run(self):

        cmd = self.actor.bcast
        #cmd.inform('guideReady=0')

        while True:

            if self.__stop.is_set():
                self.__stop.clear()
                break
            start = time.time()
            mode, target_id, exposure_time, cadence, focus = self._get_params()
            self.logger.info('AgThread.run: mode={},target_id={},exposure_time={},cadence={},focus={}'.format(mode, target_id, exposure_time, cadence, focus))
            try:
                if mode in (ag.Mode.INIT, ag.Mode.AUTO, ag.Mode.INIT_DB, ag.Mode.AUTO_DB):
                    autoguide.set_target(target_id=target_id, logger=self.logger)
                    if mode in (ag.Mode.INIT_DB, ag.Mode.AUTO_DB):
                        autoguide.set_catalog(logger=self.logger)
                        mode = ag.Mode.ON if mode == ag.Mode.AUTO_DB else ag.Mode.OFF
                        self._set_params(mode=mode)
                if mode in (ag.Mode.ON, ag.Mode.INIT, ag.Mode.AUTO):
                    result = self.actor.sendCommand(
                        actor='agcam',
                        cmdStr='expose speed={}'.format(exposure_time),
                        timeLim=(exposure_time // 1000 + 5)
                    )
                    telescope_state = [
                        t(v) for t, v in zip(
                            (int, int, float, float, float, float, int, int),
                            self.actor.models['mlp1'].keyVarDict['telescopeState'].valueList
                        )
                    ]
                    self.logger.info('AgThread.run: telescopeState={}'.format(telescope_state))
                    frame_id = int(self.actor.models['agcam'].keyVarDict['frameId'].valueList[0])
                    self.actor.logger.info('AgThread.run: frameId={}'.format(frame_id))
                    data_time = float(self.actor.models['agcam'].keyVarDict['dataTime'].valueList[0])
                    self.logger.info('AgThread.run: dataTime={}'.format(data_time))
                    # retrieve detected objects from opdb
                    if mode in (ag.Mode.INIT, ag.Mode.AUTO):
                        # store initial conditions
                        autoguide.set_catalog(frame_id=frame_id, logger=self.logger)
                        self._set_params(mode=ag.Mode.ON if mode == ag.Mode.AUTO else ag.Mode.OFF)
                    else:  # mode == ag.Mode.ON
                        cmd.inform('detectionState=1')
                        # compute guide errors
                        dalt, daz, _ = autoguide.autoguide(frame_id=frame_id, logger=self.logger)
                        cmd.inform('detectionState=0')
                        result = self.actor.sendCommand(
                            actor='mlp1',
                            cmdStr='guide azel={},{} ready=1 time={} delay=0 xy=0,0 size=0 intensity=0 flux=0'.format(daz, dalt, data_time),
                            timeLim=5
                        )
                        #cmd.inform('guideReady=1')
                        if focus:
                            # compute focus error
                            pass
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
