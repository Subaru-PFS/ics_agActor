import logging
import time
import threading


class ag(object):

    def __init__(self, actor, name, logLevel=logging.DEBUG):

        self.actor = actor
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logLevel)
        self.thread = None

        self.exposure_time = 1000 # ms
        self.cadence = 0 # ms (0 means as fast as possible)
        self.focus = False

    def __del__(self):

        self.logger.info('ag.__del__:')

    def start(self, cmd=None):

        self.logger.info('starting ag controller...')

    def stop(self, cmd=None):

        self.logger.info('stopping ag controller...')
        self.stop_ag()

    def start_ag(self, cmd=None, exposure_time=None, cadence=None, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        if self.thread is None:
            if exposure_time is not None:
                self.exposure_time = exposure_time
            if cadence is not None:
                self.cadence = cadence
            if focus is not None:
                self.focus = focus
            self.logger.info('starting ag...')
            self.thread = AgThread(actor=self.actor, logger=self.logger, exposure_time=self.exposure_time, cadence=self.cadence, focus=self.focus)
            self.thread.start()

    def stop_ag(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        if self.thread is not None:
            self.logger.info('stopping ag...')
            self.thread.stop()
            self.thread.join()
            self.thread = None


class AgThread(threading.Thread):

    def __init__(self, actor=None, logger=None, exposure_time=1000, cadence=10000, focus=False):

        super().__init__()

        self.actor = actor
        self.logger = logger
        self.exposure_time = exposure_time
        self.cadence = cadence
        self.focus = focus
        self.lock = threading.Lock()
        self.__stop = threading.Event()

    def __del__(self):

        self.logger.info('AgThread.__del__:')

    def stop(self):

        self.__stop.set()

    def get_params(self):

        with self.lock:
            return self.exposure_time, self.cadence, self.focus

    def set_exposure_time(self, exposure_time=None):

        self.logger.info('AgThread.set_exposure_time: exposure_time={}'.format(exposure_time))
        with self.lock:
            self.exposure_time = exposure_time

    def set_cadence(self, cadence=None):

        self.logger.info('AgThread.set_cadence: cadence={}'.format(cadence))
        with self.lock:
            self.cadence = cadence

    def set_focus(self, focus=None):

        self.logger.info('AgThread.set_focus: focus={}'.format(focus))
        with self.lock:
            self.focus = focus

    def run(self):

        cmd = self.actor.bcast
        cmd.inform('guideReady=0')

        while True:

            start = time.time()
            exposure_time, cadence, focus = self.get_params()
            try:
                result = self.actor.sendCommand(
                    actor='agcam',
                    cmdStr='expose speed={}'.format(exposure_time),
                    timeLim=(exposure_time // 1000 + 5)
                )
            except Exception as e:
                self.logger.error('AgThread.run: {}'.format(e))
                break
            try:
                result = self.actor.sendCommand(
                    actor='mlp1',
                    cmdStr='set_offsets xy=0,0 altaz=0,0',
                    timeLim=5
                )
            except Exception as e:
                self.logger.error('AgThread.run: {}'.format(e))
                break
            cmd.inform('guideReady=1')
            end = time.time()
            stop = self.__stop.wait(max(0, cadence / 1000 - (end - start)))
            if stop:
                break

        cmd.inform('guideReady=0')
