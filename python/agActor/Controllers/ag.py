import enum
import logging
import threading
import time


class ag:

    class Mode(enum.Enum):

        OFF = 0
        ON = 1
        INIT = 2
        AUTO = 3

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    FOCUS = False

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

        return self.thread.mode

    def start_autoguide(self, cmd=None, initialize=True, exposure_time=EXPOSURE_TIME, cadence=CADENCE, focus=FOCUS):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.AUTO if initialize else ag.Mode.ON, exposure_time=exposure_time, cadence=cadence, focus=focus)

    def initialize_autoguide(self, cmd=None, exposure_time=EXPOSURE_TIME):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.INIT, exposure_time=exposure_time)

    def stop_autoguide(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.OFF)

    def reconfigure_autoguide(self, cmd=None, exposure_time=None, cadence=None, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(exposure_time=exposure_time, cadence=cadence, focus=focus)


class AgThread(threading.Thread):

    def __init__(self, actor=None, logger=None):

        super().__init__()

        self.actor = actor
        self.logger = logger
        self.mode = ag.Mode.OFF
        self.exposure_time = ag.EXPOSURE_TIME
        self.cadence = ag.CADENCE
        self.focus = ag.FOCUS
        self.lock = threading.Lock()
        self.__abort = threading.Event()
        self.__stop = threading.Event()

    def __del__(self):

        self.logger.info('AgThread.__del__:')

    def stop(self):

        self.__stop.set()
        self.__abort.set()

    def get_params(self):

        with self.lock:
            self.__abort.clear()
            return self.mode, self.exposure_time, self.cadence, self.focus

    def set_params(self, mode=None, exposure_time=None, cadence=None, focus=None):

        with self.lock:
            if mode is not None:
                self.__abort.set()
                self.mode = mode
            if exposure_time is not None:
                self.exposure_time = exposure_time
            if cadence is not None:
                self.cadence = cadence
            if focus is not None:
                self.focus = focus

    def run(self):

        cmd = self.actor.bcast
        #cmd.inform('guideReady=0')

        while True:

            if self.__stop.is_set():
                self.__stop.clear()
                break
            start = time.time()
            mode, exposure_time, cadence, focus = self.get_params()
            self.logger.info('AgThread.run: mode={},exposure_time={},cadence={},focus={}'.format(mode, exposure_time, cadence, focus))
            try:
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
                    data_time = self.actor.models['agcam'].keyVarDict['dataTime'].valueList[0]
                    self.logger.info('AgThread.run: dataTime={}'.format(data_time))
                    # retrieve detected objects from opdb
                    if mode in (ag.Mode.INIT, ag.Mode.AUTO):
                        # store initial conditions
                        self.set_params(mode=ag.Mode.ON if mode == ag.Mode.AUTO else ag.Mode.OFF)
                    else:  # mode == ag.Mode.ON
                        # compute guide errors
                        result = self.actor.sendCommand(
                            actor='mlp1',
                            cmdStr='guide azel=0,0 ready=1 time={} delay=0 xy=0,0 size=0 intensity=0 flux=0'.format(data_time),
                            timeLim=5
                        )
                        #cmd.inform('guideReady=1')
                        if focus:
                            # compute focus error
                            pass
            except Exception as e:
                self.logger.error('AgThread.run: {}'.format(e))
            end = time.time()
            timeout = max(0, cadence / 1000 - (end - start)) if mode == ag.Mode.ON else 0.5
            self.__abort.wait(timeout)

        cmd.inform('guideReady=0')
