#!/usr/bin/env python

import time
import numpy
import opscore.protocols.keys as keys
import opscore.protocols.types as types
from agActor import field_acquisition, focus as _focus, data_utils, pfs_design


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.visit_id = None
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('show', '', self.show),
            ('acquire_field', '[<design_id>] [<design_path>] [<visit_id>] [<exposure_time>] [<guide>] [<center>] [<magnitude>]', self.acquire_field),
            ('focus', '[<visit_id>] [<exposure_time>]', self.focus),
            ('autoguide', '@start [<design_id>] [<design_path>] [<visit_id>] [<from_sky>] [<exposure_time>] [<cadence>] [<focus>]', self.start_autoguide),
            ('autoguide', '@initialize [<design_id>] [<design_path>] [<visit_id>] [<from_sky>] [<exposure_time>] [<cadence>] [<focus>]', self.initialize_autoguide),
            ('autoguide', '@restart', self.restart_autoguide),
            ('autoguide', '@stop', self.stop_autoguide),
            ('autoguide', '@reconfigure [<exposure_time>] [<cadence>] [<focus>]', self.reconfigure_autoguide),
        ]
        self.keys = keys.KeysDictionary(
            'ag_ag',
            (1, 7),
            keys.Key('exposure_time', types.Int(), help=''),
            keys.Key('cadence', types.Int(), help=''),
            keys.Key('focus', types.Bool('no', 'yes'), help=''),
            keys.Key('guide', types.Bool('no', 'yes'), help=''),
            keys.Key('design_id', types.String(), help=''),
            keys.Key('design_path', types.String(), help=''),
            keys.Key('visit_id', types.Int(), help=''),
            keys.Key('from_sky', types.Bool('no', 'yes'), help=''),
            keys.Key('center', types.Float() * (2, 3), help=''),
            keys.Key('magnitude', types.Float(), help=''),
        )
        self.with_opdb_agc_guide_offset = actor.config.getboolean(actor.name, 'agc_guide_offset', fallback=False)
        self.with_opdb_agc_match = actor.config.getboolean(actor.name, 'agc_match', fallback=False)
        self.with_agcc_timestamp = actor.config.getboolean(actor.name, 'agcc_timestamp', fallback=False)
        tel_status = [x.strip() for x in actor.config.get(actor.name, 'tel_status', fallback='agc_exposure').split(',')]
        self.with_gen2_status = 'gen2' in tel_status
        self.with_mlp1_status = 'mlp1' in tel_status
        self.with_opdb_tel_status = 'tel_status' in tel_status

    def ping(self, cmd):
        """Return a product name."""

        cmd.inform('text="{}"'.format(self.actor.productName))
        cmd.finish()

    def status(self, cmd):
        """Return status keywords."""

        self.actor.sendVersionKey(cmd)
        #self.actor.ag.sendStatusKeys(cmd, force=True)
        cmd.finish()

    def show(self, cmd):
        """Show status keywords from all models."""

        for n in self.actor.models:
            try:
                d = self.actor.models[n].keyVarDict
                for k, v in d.items():
                    cmd.inform('text="{}"'.format(repr(v)))
            except Exception as e:
                cmd.warn('text="AgCmd.show: {}: {}"'.format(n, e))
        cmd.finish()

    def acquire_field(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))
        mode = controller.get_mode()
        if mode != controller.Mode.OFF:
            cmd.fail('text="AgCmd.acquire_field: mode={}'.format(mode))
            return

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        guide = False
        if 'guide' in cmd.cmd.keywords:
            guide = bool(cmd.cmd.keywords['guide'].values[0])
        center = None
        if 'center' in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords['center'].values])
        magnitude = None
        if 'magnitude' in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords['magnitude'].values[0])

        try:
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
                    self.actor.logger.info('AgCmd.acquire_field: tel_status={}'.format(tel_status))
                    kwargs['tel_status'] = tel_status
                    if center is None:
                        center = tel_status[5:8]
                if self.with_opdb_tel_status:
                    status_update = self.actor.gen2.statusUpdate
                    status_id = (status_update['visit'], status_update['sequence'])
                    self.actor.logger.info('AgCmd.acquire_field: status_id={}'.format(status_id))
                    kwargs['status_id'] = status_id
            telescope_state = None
            if self.with_mlp1_status:
                telescope_state = self.actor.mlp1.telescopeState
                self.actor.logger.info('AgCmd.acquire_field: telescopeState={}'.format(telescope_state))
            # wait for an exposure to complete
            result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info('AgCmd.acquire_field: frameId={}'.format(frame_id))
            data_time = self.actor.agcc.dataTime
            self.actor.logger.info('AgCmd.acquire_field: dataTime={}'.format(data_time))
            if self.with_agcc_timestamp:
                kwargs['taken_at'] = data_time  # unix timestamp, not timezone-aware datetime
            if center is not None:
                kwargs['center'] = center
            if magnitude is not None:
                kwargs['magnitude'] = magnitude
            # retrieve field center coordinates from opdb
            # retrieve exposure information from opdb
            # retrieve guide star coordinates from opdb
            # retrieve metrics of detected objects from opdb
            # compute offsets, scale, transparency, and seeing
            dalt = daz = None
            if guide:
                cmd.inform('detectionState=1')
                # convert equatorial coordinates to horizontal coordinates
                ra, dec, pa, dra, ddec, dinr, dalt, daz, *values = field_acquisition.acquire_field(design=design, frame_id=frame_id, altazimuth=True, logger=self.actor.logger, **kwargs)
                cmd.inform('text="dra={},ddec={},dinr={},dalt={},daz={}"'.format(dra, ddec, dinr, dalt, daz))
                filenames = ('/dev/shm/guide_objects.npy', '/dev/shm/detected_objects.npy', '/dev/shm/identified_objects.npy')
                for filename, value in zip(filenames, values):
                    numpy.save(filename, value)
                cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, pa, *filenames))
                cmd.inform('detectionState=0')
                dx, dy, size, peak, flux = values[3], values[4], values[5], values[6], values[7]
                # send corrections to mlp1 and gen2 (or iic)
                result = self.actor.queueCommand(
                    actor='mlp1',
                    # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
                    cmdStr='guide azel={},{} ready=1 time={} delay=0 xy={},{} size={} intensity={} flux={}'.format(- daz, - dalt, data_time, dx / 98e-6, - dy / 98e-6, size * 13 / 98e-3, peak, flux),
                    timeLim=5
                )
                result.get()
                #cmd.inform('guideReady=1')
            else:
                cmd.inform('detectionState=1')
                ra, dec, pa, dra, ddec, dinr, *values = field_acquisition.acquire_field(design=design, frame_id=frame_id, logger=self.actor.logger, **kwargs)
                cmd.inform('text="dra={},ddec={},dinr={}"'.format(dra, ddec, dinr))
                filenames = ('/dev/shm/guide_objects.npy', '/dev/shm/detected_objects.npy', '/dev/shm/identified_objects.npy')
                for filename, value in zip(filenames, values):
                    numpy.save(filename, value)
                cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, pa, *filenames))
                cmd.inform('detectionState=0')
                # send corrections to gen2 (or iic)
            # store results in opdb
            if self.with_opdb_agc_guide_offset:
                # always compute focus offset and tilt
                dz, dzs = _focus._focus(detected_objects=values[1], logger=self.actor.logger)
                data_utils.write_agc_guide_offset(
                    frame_id=frame_id,
                    ra=ra,
                    dec=dec,
                    pa=pa,
                    delta_ra=dra,
                    delta_dec=ddec,
                    delta_insrot=dinr,
                    delta_az=daz,
                    delta_el=dalt,
                    delta_z=dz,
                    delta_zs=dzs
                )
            if self.with_opdb_agc_match:
                data_utils.write_agc_match(
                    design_id=design_id if design_id is not None else pfs_design.to_design_id(design_path) if design_path is not None else 0,
                    frame_id=frame_id,
                    guide_objects=values[0],
                    detected_objects=values[1],
                    identified_objects=values[2]
                )
        except Exception as e:
            cmd.fail('text="AgCmd.acquire_field: {}'.format(e))
            return
        cmd.finish()

    def focus(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))
        mode = controller.get_mode()
        if mode != controller.Mode.OFF:
            cmd.fail('text="AgCmd.focus: mode={}'.format(mode))
            return

        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100

        try:
            cmd.inform('exposureTime={}'.format(exposure_time))
            # start an exposure
            result = self.actor.queueCommand(
                actor='agcc',
                cmdStr='expose object pfsVisitId={} exptime={} centroid=1'.format(visit_id, exposure_time / 1000),
                timeLim=(exposure_time // 1000 + 5)
            )
            # wait for an exposure to complete
            result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info('AgCmd.focus: frameId={}'.format(frame_id))
            # retrieve detected objects from agcc (or opdb)
            # compute focus offset and tilt
            dz, dzs = _focus.focus(frame_id=frame_id, logger=self.actor.logger)
            if numpy.isnan(dz):
                cmd.fail('text="AgCmd.focus: dz={}'.format(dz))
                return
            cmd.inform('text="dz={}"'.format(dz))
            # send corrections to gen2 (or iic)
            # store results in opdb
            if self.with_opdb_agc_guide_offset:
                data_utils.write_agc_guide_offset(
                    frame_id=frame_id,
                    delta_z=dz,
                    delta_zs=dzs
                )
        except Exception as e:
            cmd.fail('text="AgCmd.focus: {}'.format(e))
            return
        cmd.finish()

    def start_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        design = None if all(x is None for x in (design_id, design_path)) else (design_id, design_path)
        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        from_sky = None
        if 'from_sky' in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords['from_sky'].values[0])
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = 0  # ms
        if 'cadence' in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords['cadence'].values[0])
            if cadence < 0:
                cadence = 0
        focus = False
        if 'focus' in cmd.cmd.keywords:
            focus = bool(cmd.cmd.keywords['focus'].values[0])

        try:
            controller.start_autoguide(cmd=cmd, design=design, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.start_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        if all(x is None for x in (design_id, design_path)):
            cmd.fail('text="AgCmd.initialize_autoguide: <design_id> and/or <design_path> required"')
            return
        design = (design_id, design_path)
        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        from_sky = None
        if 'from_sky' in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords['from_sky'].values[0])
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = 0  # ms
        if 'cadence' in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords['cadence'].values[0])
            if cadence < 0:
                cadence = 0
        focus = False
        if 'focus' in cmd.cmd.keywords:
            focus = bool(cmd.cmd.keywords['focus'].values[0])

        try:
            controller.initialize_autoguide(cmd=cmd, design=design, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.initialize_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def restart_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        try:
            controller.start_autoguide(cmd=cmd, exposure_time=None, cadence=None, focus=None)
        except Exception as e:
            cmd.fail('text="AgCmd.restart_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def stop_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        try:
            controller.stop_autoguide()
        except Exception as e:
            cmd.fail('text="AgCmd.stop_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def reconfigure_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        exposure_time = None
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = None
        if 'cadence' in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords['cadence'].values[0])
            if cadence < 0:
                cadence = 0
        focus = None
        if 'focus' in cmd.cmd.keywords:
            focus = bool(cmd.cmd.keywords['focus'].values[0])

        try:
            controller.reconfigure_autoguide(cmd=cmd, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.reconfigure_autoguide: {}"'.format(e))
            return
        cmd.finish()
