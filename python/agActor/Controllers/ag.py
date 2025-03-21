import enum
import logging
import threading
import time

import numpy as np

from agActor import autoguide, data_utils, focus, pfs_design
from agActor.telescope_center import telCenter as tel_center
from agActor.utils import FILENAMES


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
        AUTO_OTF = REF_OTF | ON  # [-IS] auto-start, guide objects from catalog on the fly
        # AUTO_ONCE_SKY = REF_SKY | ONCE  # [-IS] initialize and autoguide once, guide objects from exposure
        AUTO_ONCE_DB = REF_DB | ONCE  # [-IS] initialize and autoguide once, guide objects from opdb
        #       (field acquisition with autoguider)
        AUTO_ONCE_OTF = REF_OTF | ONCE  # [-IS] initialize and autoguide once, guide objects from catalog on the fly
        #       (on-the-fly field acquisition with autoguider)

    INITIAL = False
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
    EXPOSURE_DELAY = 100  # ms
    TEC_OFF = False

    class Params:

        __slots__ = ('mode', 'design', 'visit_id', 'exposure_time', 'cadence', 'center', 'options')

        _OPTIONS = ('catalog', 'magnitude', 'dry_run', 'fit_dinr', 'fit_dscale',
                    'max_ellipticity', 'max_size', 'min_size',
                    'max_residual', 'exposure_delay', 'tec_off')

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

    def start_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME,
                        cadence=CADENCE, center=None, **kwargs
                        ):

        # cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO_SKY if from_sky else ag.Mode.AUTO_DB if design is not None else ag.Mode.AUTO_OTF
        self.thread.set_params(
            mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, cadence=cadence, center=center,
            options={}, **kwargs
        )

    def restart_autoguide(self, cmd=None):

        # cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.ON
        self.thread.set_params(mode=mode)

    def initialize_autoguide(self, cmd=None, design=None, visit_id=None, from_sky=None, exposure_time=EXPOSURE_TIME,
                             cadence=CADENCE, center=None, **kwargs
                             ):

        # cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.REF_SKY if from_sky else ag.Mode.REF_DB if design is not None else ag.Mode.REF_OTF
        self.thread.set_params(
            mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, cadence=cadence, center=center,
            options={}, **kwargs
        )

    def stop_autoguide(self, cmd=None):

        # cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(mode=ag.Mode.STOP)

    def reconfigure_autoguide(self, cmd=None, **kwargs):

        # cmd = cmd if cmd else self.actor.bcast
        self.thread.set_params(**kwargs)

    def acquire_field(self, cmd=None, design=None, visit_id=None, exposure_time=EXPOSURE_TIME, center=None, **kwargs):

        # cmd = cmd if cmd else self.actor.bcast
        mode = ag.Mode.AUTO_ONCE_DB if design is not None else ag.Mode.AUTO_ONCE_OTF
        self.thread.set_params(
            mode=mode, design=design, visit_id=visit_id, exposure_time=exposure_time, center=center, options={},
            **kwargs
        )


class AgThread(threading.Thread):

    def __init__(self, actor=None, logger=None):

        super().__init__()

        self.actor = actor
        self.logger = logger
        self.input_params = {}
        self.params = ag.Params(mode=ag.Mode.OFF, exposure_time=ag.EXPOSURE_TIME, cadence=ag.CADENCE)
        self.lock = threading.Lock()
        self.__abort = threading.Event()
        self.__stop = threading.Event()

        self.with_opdb_agc_guide_offset = actor.actorConfig.get('agc_guide_offset', False)
        self.with_opdb_agc_match = actor.actorConfig.get('agc_match', False)
        self.with_agcc_timestamp = actor.actorConfig.get('agcc_timestamp', False)
        tel_status = [x.strip() for x in actor.actorConfig.get('tel_status', ('agc_exposure',)).split(',')]
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

            start_time = time.time()
            mode, design, visit_id, exposure_time, cadence, center, options = self._get_params()
            design_id, design_path = design if design is not None else (None, None)

            self.logger.info(
                f"AgThread.run: "
                f"mode={mode},design={design},visit_id={visit_id},"
                f"exposure_time={exposure_time},cadence={cadence},"
                f"center={center},options={options}"
            )
            dither, offset = None, None

            try:
                # If we are running On-The-Fly (OTF) mode, we need to get info from the telescope.
                if mode & ag.Mode.REF_OTF and not mode & (ag.Mode.ON | ag.Mode.ONCE):
                    if self.with_gen2_status:
                        kwargs = dict()
                        if self.with_mlp1_status:
                            telescope_state = self.actor.mlp1.telescopeState
                            self.logger.info(f"AgThread.run: telescopeState={telescope_state}")
                            kwargs['inr'] = telescope_state['rotator_real_angle']

                        # update gen2 status values
                        self.actor.queueCommand(
                            actor='gen2',
                            cmdStr=f"updateTelStatus caller={self.actor.name}",
                            timeLim=5
                        ).get()

                        tel_status = self.actor.gen2.tel_status
                        self.logger.info(f"AgThread.run: tel_status={tel_status}")

                        kwargs['tel_status'] = tel_status
                        _tel_center = tel_center(actor=self.actor, center=center, design=design, tel_status=tel_status)

                        if center is None:
                            # dithered center and guide offset correction (insrot only)
                            center, offset = _tel_center.dither
                            self.logger.info(f"AgThread.run: center={center}")
                        else:
                            # dithering and guide offset correction
                            offset = _tel_center.offset

                        self.logger.info(f"AgThread.run: offset={offset}")

                        if self.with_mlp1_status:
                            telescope_state = self.actor.mlp1.telescopeState
                            taken_at = self.actor.mlp1.setUnixDay(
                                telescope_state['az_el_detect_time'], tel_status[8].timestamp()
                            )
                            kwargs['taken_at'] = taken_at

                        if center is not None:
                            kwargs['center'] = center

                        if offset is not None:
                            kwargs['offset'] = offset

                        kwargs['catalog'] = options.get('catalog', ag.CATALOG)
                        kwargs['magnitude'] = options.get('magnitude', ag.MAGNITUDE)

                        autoguide.set_field(logger=self.logger, **kwargs)

                        # Turn off On-The-Fly mode.
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)

                # If we are using values from the DB.
                if mode & ag.Mode.REF_DB:
                    kwargs = {
                        'catalog': options.get('catalog', ag.CATALOG),
                        'magnitude': options.get('magnitude', ag.MAGNITUDE),
                    }

                    autoguide.set_field(design=design, logger=self.logger, **kwargs)

                    # Turn off REF_DB mode.
                    mode &= ~ag.Mode.REF_DB
                    self._set_params(mode=mode)

                # If we are running in a mode that requires an exposure.
                if mode & (ag.Mode.ON | ag.Mode.ONCE | ag.Mode.REF_SKY):
                    exposure_delay = options.get('exposure_delay', ag.EXPOSURE_DELAY)
                    tec_off = options.get('tec_off', ag.TEC_OFF)
                    cmd.inform(f"exposureTime={exposure_time}")

                    # start an exposure
                    cmdStr = f"expose object exptime={exposure_time / 1000} centroid=1"
                    if visit_id is not None:
                        cmdStr += f" visit={visit_id}"
                    if exposure_delay > 0:
                        cmdStr += f" threadDelay={exposure_delay}"
                    if tec_off:
                        cmdStr += ' tecOFF'

                    # Take the actual exposure.
                    result = self.actor.queueCommand(
                        actor='agcc',
                        cmdStr=cmdStr,
                        timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 5)
                    )

                    # Wait for the exposure.
                    time.sleep((exposure_time + 7 * exposure_delay) / 1000 / 2)

                    kwargs = {}
                    telescope_state = None

                    if self.with_mlp1_status:
                        telescope_state = self.actor.mlp1.telescopeState
                        self.logger.info(f"AgThread.run: telescopeState={telescope_state}")
                        kwargs['inr'] = telescope_state['rotator_real_angle']

                    # Update gen2 status values
                    if self.with_gen2_status or self.with_opdb_tel_status:
                        self.actor.queueCommand(
                            actor='gen2',
                            cmdStr=f"updateTelStatus caller={self.actor.name}",
                            timeLim=5
                        ).get()

                        if self.with_gen2_status:
                            tel_status = self.actor.gen2.tel_status
                            self.logger.info(f"AgThread.run: tel_status={tel_status}")
                            kwargs['tel_status'] = tel_status
                            _tel_center = tel_center(
                                actor=self.actor, center=center, design=design, tel_status=tel_status
                            )
                            if all(x is None for x in (center, design)):
                                center, offset = _tel_center.dither  # dithered center and guide offset correction (
                                # insrot only)
                                self.logger.info(f"AgThread.run: center={center}")
                            else:
                                offset = _tel_center.offset  # dithering and guide offset correction
                            self.logger.info(f"AgThread.run: offset={offset}")

                        if self.with_opdb_tel_status:
                            status_update = self.actor.gen2.statusUpdate
                            status_id = (status_update['visit'], status_update['sequenceNum'])
                            self.logger.info(f"AgThread.run: status_id={status_id}")
                            kwargs['status_id'] = status_id

                    # wait for an exposure to complete.
                    result.get()

                    # Do things with the exposure.
                    frame_id = self.actor.agcc.frameId
                    data_time = self.actor.agcc.dataTime
                    taken_at = data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2

                    self.logger.info(f"AgThread.run: frameId={frame_id}")
                    self.logger.info(f"AgThread.run: dataTime={data_time}")
                    self.logger.info(f"AgThread.run: taken_at={taken_at}")

                    if self.with_agcc_timestamp:
                        # unix timestamp, not timezone-aware datetime
                        kwargs['taken_at'] = taken_at

                    if self.with_mlp1_status:
                        # possibly override timestamp from agcc
                        taken_at = self.actor.mlp1.setUnixDay(telescope_state['az_el_detect_time'], taken_at)
                        kwargs['taken_at'] = taken_at

                    if center is not None:
                        kwargs['center'] = center

                    if offset is not None:
                        kwargs['offset'] = offset

                    kwargs['catalog'] = options.get('catalog', ag.CATALOG)
                    kwargs['magnitude'] = options.get('magnitude', ag.MAGNITUDE)

                    if mode & ag.Mode.REF_OTF:
                        autoguide.set_field(logger=self.logger, **kwargs)

                        # Turn off On-The-Fly mode.
                        mode &= ~ag.Mode.REF_OTF
                        self._set_params(mode=mode)

                    if mode & ag.Mode.REF_SKY:
                        # store initial conditions
                        # center takes precedence over design
                        autoguide.set_field(design=design, frame_id=frame_id, logger=self.logger, **kwargs)

                        # Turn off REF_SKY mode.
                        mode &= ~ag.Mode.REF_SKY
                        self._set_params(mode=mode)
                    else:  # mode & (ag.Mode.ON | ag.Mode.ONCE)
                        dry_run = options.get('dry_run', ag.DRY_RUN)

                        if 'fit_dinr' in options:
                            kwargs['fit_dinr'] = options.get('fit_dinr')
                        if 'fit_dscale' in options:
                            kwargs['fit_dscale'] = options.get('fit_dscale')
                        if 'max_ellipticity' in options:
                            kwargs['max_ellipticity'] = options.get('max_ellipticity')
                        if 'max_size' in options:
                            kwargs['max_size'] = options.get('max_size')
                        if 'min_size' in options:
                            kwargs['min_size'] = options.get('min_size')
                        if 'max_residual' in options:
                            kwargs['max_residual'] = options.get('max_residual')

                        cmd.inform('detectionState=1')

                        # compute guide errors
                        offset_info = autoguide.acquire_field(frame_id=frame_id, logger=self.logger, **kwargs)

                        # TODO: remove these and use offset_info directly.
                        ra = offset_info.ra
                        dec = offset_info.dec
                        inst_pa = offset_info.inst_pa
                        dra = offset_info.dra
                        ddec = offset_info.ddec
                        dinr = offset_info.dinr
                        dscale = offset_info.dscale
                        dalt = offset_info.dalt
                        daz = offset_info.daz
                        dx = offset_info.dx
                        dy = offset_info.dy
                        spot_size = offset_info.spot_size
                        peak_intensity = offset_info.peak_intensity
                        flux = offset_info.flux

                        # Save the detected, guide, and identified objects.
                        for save_name, path in FILENAMES.items():
                            np.save(path, getattr(offset_info, save_name))

                        cmd.inform(f'text="{ra=},{dec=},{inst_pa=},{dra=},{ddec=},{dinr=},{dscale=},{dalt=},{daz=}"')
                        cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, inst_pa, *FILENAMES.values()))
                        cmd.inform('detectionState=0')

                        # Send the guide commands to the telescope.
                        result = self.actor.queueCommand(
                            actor='mlp1',
                            cmdStr=f"guide "
                                   f"azel={-daz},{-dalt} "
                                   f"ready={int(not dry_run)} "
                                   f"time={taken_at} "
                                   f"delay=0 "
                                   f"xy={dx * 1e3},{-dy * 1e3} "
                                   f"size={spot_size * 13 / 98e-3} "
                                   f"intensity={peak_intensity} "
                                   f"flux={flux}",
                            timeLim=5
                        )
                        # Wait for result of guiding command.
                        result.get()
                        # cmd.inform('guideReady=1')

                        # always compute focus offset and tilt
                        kwargs = {
                            key: kwargs.get(key)
                            for key in ('max_ellipticity', 'max_size', 'min_size')
                            if key in kwargs
                        }
                        dz, dzs = focus._focus(
                            detected_objects=offset_info.detected_objects, logger=self.logger, **kwargs
                        )

                        # send corrections to gen2 (or iic)
                        cmd.inform(f"guideErrors={frame_id},{dra},{ddec},{dinr},{daz},{dalt},{dz},{dscale}")
                        cmd.inform(f"focusErrors={frame_id},{dzs[0]},{dzs[1]},{dzs[2]},{dzs[3]},{dzs[4]},{dzs[5]}")

                        # store results in opdb.
                        if self.with_opdb_agc_guide_offset:
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
                                delta_zs=dzs
                            )

                        if self.with_opdb_agc_match:
                            design_id = design_id or 0
                            if design_path is not None and design_id == 0:
                                design_id = pfs_design.pfsDesign.to_design_id(design_path)

                            data_utils.write_agc_match(
                                design_id=design_id,
                                frame_id=frame_id,
                                guide_objects=offset_info.guide_objects,
                                detected_objects=offset_info.detected_objects,
                                identified_objects=offset_info.identified_objects
                            )

                # If we are running once, we need to stop the autoguider.
                if mode & ag.Mode.ONCE:
                    self._set_params(mode=ag.Mode.OFF)

                # If we are running in stop mode, we need to stop the autoguider.
                if mode == ag.Mode.STOP:
                    # cmd.inform('detectionState=0')
                    cmd.inform('guideReady=0')
                    self._set_params(mode=ag.Mode.OFF)
            except Exception as e:
                self.logger.exception('AgThread.run:')

            end_time = time.time()
            timeout = max(0, cadence / 1000 - (end_time - start_time)) if mode == ag.Mode.ON else 0.5
            self.__abort.wait(timeout)

        # Run loop has stopped, marker the guide as not ready.
        cmd.inform('guideReady=0')
