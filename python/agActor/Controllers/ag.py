import enum
import logging
import threading
import time
import numpy
from agActor import autoguide, focus as _focus, data_utils, pfs_design


class ag:

    class Mode(enum.IntFlag):

        # flags (Fs), inputs (Is), and states (Ss)
        OFF = 0                         # [--S] idle
        ON = 1                          # [FIS] start/resume autoguide
        ONCE = 2                        # [FIS] autoguide once
        REF_SKY = 4                     # [FIS] initialize only, guide objects from exposure
        REF_DB = 8                      # [FIS] initialize only, guide objects from opdb
        REF_OTF = 16                    # [FIS] initialize only, guide objects from catalog on the fly
        STOP = 32                       # [-I-] stop autoguide
        AUTO_SKY = REF_SKY | ON         # [-IS] auto-start, guide objects from first exposure
        AUTO_DB = REF_DB | ON           # [-IS] auto-start, guide objects from opdb
        AUTO_OTF = REF_OTF | ON         # [-IS] auto-start, guide objects from catalog on the fly
        #AUTO_ONCE_SKY = REF_SKY | ONCE  # [-IS] initialize and autoguide once, guide objects from exposure
        AUTO_ONCE_DB = REF_DB | ONCE    # [-IS] initialize and autoguide once, guide objects from opdb
                                        #       (field acquisition with autoguider)
        AUTO_ONCE_OTF = REF_OTF | ONCE  # [-IS] initialize and autoguide once, guide objects from catalog on the fly
                                        #       (on-the-fly field acquisition with autoguider)

    EXPOSURE_TIME = 2000  # ms
    CADENCE = 0  # ms
    FOCUS = False
    MAGNITUDE = 20.0

    class Params:

        _KEYS = ('mode', 'design', 'visit_id', 'exposure_time', 'cadence', 'focus', 'center', 'magnitude')

        def __init__(self, **kwargs):

            self.reset(**kwargs)

        def reset(self, **kwargs):

            for key in ag.Params._KEYS:
                setattr(self, key, kwargs.get(key))

        def set(self, **kwargs):

            for key in kwargs:
                if key in ag.Params._KEYS:
                    setattr(self, key, kwargs.get(key))

        def get(self):

            return tuple([getattr(self, key, None) for key in ag.Params._KEYS])

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

        mode, *_ = self.thread.get_params()
        return mode

    def start_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME, cadence=CADENCE, focus=FOCUS, center=None, magnitude=MAGNITUDE):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO_SKY if from_sky else ag.Mode.AUTO_DB if design is not None else ag.Mode.AUTO_OTF
        self.thread.set_params(mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, cadence=cadence, focus=focus, center=center, magnitude=magnitude)

    def restart_autoguide(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.ON
        self.thread.set_params(mode=mode)

    def initialize_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME, cadence=CADENCE, focus=FOCUS, center=None, magnitude=MAGNITUDE):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.REF_SKY if from_sky else ag.Mode.REF_DB if design is not None else ag.Mode.REF_OTF
        self.thread.set_params(mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, cadence=cadence, focus=focus, center=center, magnitude=magnitude)

    def stop_autoguide(self, cmd=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.STOP)

    def reconfigure_autoguide(self, cmd=None, exposure_time=None, cadence=None, focus=None):

        #cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(exposure_time=exposure_time, cadence=cadence, focus=focus)

    def acquire_field(self, cmd=None, design=None, visit_id=None, exposure_time=EXPOSURE_TIME, focus=None, center=None, magnitude=MAGNITUDE):

        #cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO_ONCE_DB if design is not None else ag.Mode.AUTO_ONCE_OTF
        self.thread.set_params(mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, focus=focus, center=center, magnitude=magnitude)


class AgThread(threading.Thread):

    def __init__(self, actor=None, logger=None):

        super().__init__()

        self.actor = actor
        self.logger = logger
        self.input_params = {}
        self.params = ag.Params(mode=ag.Mode.OFF, exposure_time=ag.EXPOSURE_TIME, cadence=ag.CADENCE, focus=ag.FOCUS, magnitude=ag.MAGNITUDE)
        self.lock = threading.Lock()
        self.__abort = threading.Event()
        self.__stop = threading.Event()

        self.with_opdb_agc_guide_offset = actor.config.getboolean(actor.name, 'agc_guide_offset', fallback=False)
        self.with_opdb_agc_match = actor.config.getboolean(actor.name, 'agc_match', fallback=False)
        self.with_agcc_timestamp = actor.config.getboolean(actor.name, 'agcc_timestamp', fallback=False)
        tel_status = [x.strip() for x in actor.config.get(actor.name, 'tel_status', fallback='agc_exposure').split(',')]
        self.with_gen2_status = 'gen2' in tel_status
        self.with_mlp1_status = 'mlp1' in tel_status
        self.with_opdb_tel_status = 'tel_status' in tel_status

    def __del__(self):

        self.logger.info('AgThread.__del__:')

    def stop(self):

        self.__stop.set()
        self.__abort.set()

    def _get_params(self):

        with self.lock:
            self.__abort.clear()
            self.params.set(**self.input_params)
            self.input_params.clear()
            return self.params.get()

    def _set_params(self, **kwargs):

        with self.lock:
            self.params.set(**kwargs)

    def get_params(self):

        with self.lock:
            return self.params.get()

    def set_params(self, **kwargs):

        with self.lock:
            self.input_params.update(**kwargs)
            if 'mode' in kwargs:
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
            mode, design, visit_id, exposure_time, cadence, focus, center, magnitude = self._get_params()
            design_id, design_path = design if design is not None else (None, None)
            self.logger.info('AgThread.run: mode={},design={},visit_id={},exposure_time={},cadence={},focus={},center={},magnitude={}'.format(mode, design, visit_id, exposure_time, cadence, focus, center, magnitude))
            try:
                if mode & ag.Mode.REF_OTF:
                    # field center coordinates need to be specified in order to be able to select guide objects on the fly at this point
                    if center is None:
                        # if an exposure is not going to be taken, telescope status values need to be queried at this point
                        if not mode & (ag.Mode.ON | ag.Mode.ONCE):
                            if self.with_gen2_status:
                                # update gen2 status values
                                self.actor.queueCommand(
                                    actor='gen2',
                                    cmdStr='updateTelStatus caller={}'.format(self.actor.name) if self.with_opdb_tel_status else 'updateTelStatus',
                                    timeLim=5
                                ).get()
                                tel_status = self.actor.gen2.tel_status
                                self.logger.info('AgThread.run: tel_status={}'.format(tel_status))
                                center = tel_status[5:8]
                                self.logger.info('AgThread.run: center={}'.format(center))
                    if center is not None:
                        autoguide.set_design(logger=self.logger, center=center)
                        kwargs = {}
                        if magnitude is not None:
                            kwargs['magnitude'] = magnitude
                        autoguide.set_design_agc(logger=self.logger, **kwargs)
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)
                if mode & ag.Mode.REF_DB:
                    autoguide.set_design(design=design, logger=self.logger)
                    kwargs = {}
                    if magnitude is not None:
                        kwargs['magnitude'] = magnitude
                    autoguide.set_design_agc(logger=self.logger, **kwargs)  # obstime=<current time>
                    mode &= ~ag.Mode.REF_DB
                    self._set_params(mode=mode)
                if mode & (ag.Mode.ON | ag.Mode.ONCE | ag.Mode.REF_SKY):
                    cmd.inform('exposureTime={}'.format(exposure_time))
                    # start an exposure
                    result = self.actor.queueCommand(
                        actor='agcc',
                        cmdStr='expose object pfsVisitId={} exptime={} centroid=1'.format(visit_id, exposure_time / 1000),
                        timeLim=(exposure_time // 1000 + 5)
                    )
                    kwargs = {}
                    if self.with_gen2_status or self.with_opdb_tel_status:
                        # update gen2 status values
                        time.sleep(exposure_time / 1000 / 2)
                        self.actor.queueCommand(
                            actor='gen2',
                            cmdStr='updateTelStatus caller={}'.format(self.actor.name) if self.with_opdb_tel_status else 'updateTelStatus',
                            timeLim=5
                        ).get()
                        if self.with_gen2_status:
                            tel_status = self.actor.gen2.tel_status
                            self.logger.info('AgThread.run: tel_status={}'.format(tel_status))
                            kwargs['tel_status'] = tel_status
                            if center is None:
                                center = tel_status[5:8]
                                self.logger.info('AgThread.run: center={}'.format(center))
                        if self.with_opdb_tel_status:
                            status_update = self.actor.gen2.statusUpdate
                            status_id = (status_update['visit'], status_update['sequence'])
                            self.logger.info('AgThread.run: status_id={}'.format(status_id))
                            kwargs['status_id'] = status_id
                    telescope_state = None
                    if self.with_mlp1_status:
                        telescope_state = self.actor.mlp1.telescopeState
                        self.logger.info('AgThread.run: telescopeState={}'.format(telescope_state))
                    # wait for an exposure to complete
                    result.get()
                    frame_id = self.actor.agcc.frameId
                    self.logger.info('AgThread.run: frameId={}'.format(frame_id))
                    data_time = self.actor.agcc.dataTime
                    self.logger.info('AgThread.run: dataTime={}'.format(data_time))
                    if self.with_agcc_timestamp:
                        kwargs['taken_at'] = data_time  # unix timestamp, not timezone-aware datetime
                    if mode & ag.Mode.REF_OTF:
                        if center is not None:
                            kwargs['center'] = center
                        if magnitude is not None:
                            kwargs['magnitude'] = magnitude
                        autoguide.set_design(logger=self.logger, **kwargs)
                        autoguide.set_design_agc(logger=self.logger, **kwargs)
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)
                    # retrieve detected objects from opdb
                    if mode & ag.Mode.REF_SKY:
                        # store initial conditions
                        autoguide.set_design(design=design, logger=self.logger, center=center)  # center takes precedence over design
                        autoguide.set_design_agc(frame_id=frame_id, logger=self.logger, **kwargs)
                        mode &= ~ag.Mode.REF_SKY
                        self._set_params(mode=mode)
                    else:  # mode & (ag.Mode.ON | ag.Mode.ONCE)
                        cmd.inform('detectionState=1')
                        # compute guide errors
                        dalt, daz, dinr, *values = autoguide.autoguide(frame_id=frame_id, logger=self.logger, **kwargs)
                        ra, dec, pa = autoguide.Field.center
                        filenames = ('/dev/shm/guide_objects.npy', '/dev/shm/detected_objects.npy', '/dev/shm/identified_objects.npy')
                        for filename, value in zip(filenames, values):
                            numpy.save(filename, value)
                        cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, pa, *filenames))
                        cmd.inform('detectionState=0')
                        dx, dy, size, peak, flux = values[3], values[4], values[5], values[6], values[7]
                        result = self.actor.queueCommand(
                            actor='mlp1',
                            # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
                            cmdStr='guide azel={},{} ready=1 time={} delay=0 xy={},{} size={} intensity={} flux={}'.format(- daz, - dalt, data_time, dx / 98e-6, - dy / 98e-6, size * 13 / 98e-3, peak, flux),
                            timeLim=5
                        )
                        result.get()
                        #cmd.inform('guideReady=1')
                        # always compute focus offset and tilt
                        dz, dzs = _focus._focus(detected_objects=values[1], logger=self.logger)
                        if focus:
                            # send corrections to gen2 (or iic)
                            pass
                        if self.with_opdb_agc_guide_offset:
                            data_utils.write_agc_guide_offset(
                                frame_id=frame_id,
                                ra=ra,
                                dec=dec,
                                pa=pa,
                                delta_insrot=dinr,
                                delta_az=daz,
                                delta_el=dalt,
                                delta_z=dz,
                                delta_zs=dzs
                            )
                        if self.with_opdb_agc_match:
                            data_utils.write_agc_match(
                                design_id=design_id if design_id is not None else pfs_design.pfsDesign.to_design_id(design_path) if design_path is not None else 0,
                                frame_id=frame_id,
                                guide_objects=values[0],
                                detected_objects=values[1],
                                identified_objects=values[2]
                            )
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
