#!/usr/bin/env python

import numpy
import opscore.protocols.keys as keys
import opscore.protocols.types as types
from agActor import field_acquisition, focus
from agActor.opdb import opDB as opdb


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.visit_id = None
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('show', '', self.show),
            ('acquire_field', '(<design_id>|<tile_id>) [<visit_id>] [<exposure_time>] [<guide>]', self.acquire_field),
            ('focus', '[<visit_id>] [<exposure_time>]', self.focus),
            ('autoguide', '@start [(<design_id>|<tile_id>)] [<visit_id>] [<from_sky>] [<exposure_time>] [<cadence>] [<focus>]', self.start_autoguide),
            ('autoguide', '@initialize (<design_id>|<tile_id>) [<visit_id>] [<from_sky>] [<exposure_time>]', self.initialize_autoguide),
            ('autoguide', '@stop', self.stop_autoguide),
            ('autoguide', '@reconfigure [<exposure_time>] [<cadence>] [<focus>]', self.reconfigure_autoguide),
        ]
        self.keys = keys.KeysDictionary(
            'ag_ag',
            (1, 4),
            keys.Key('exposure_time', types.Int(), help=''),
            keys.Key('cadence', types.Int(), help=''),
            keys.Key('focus', types.Bool('no', 'yes'), help=''),
            keys.Key('guide', types.Bool('no', 'yes'), help=''),
            keys.Key('design_id', types.Int(), help=''),
            keys.Key('tile_id', types.Int(), help=''),
            keys.Key('visit_id', types.Int(), help=''),
            keys.Key('from_sky', types.Bool('no', 'yes'), help=''),
        )

    def ping(self, cmd):
        """Return a product name."""

        cmd.inform('text="{}"'.format(self.actor.productName))
        cmd.finish()

    def status(self, cmd):
        """Return status keywords."""

        self.actor.sendVersionKey(cmd)
        # self.actor.ag.sendStatusKeys(cmd, force=True)
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

        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0])
            # get tile_id by design_id
            tile_id, *_ = opdb.query_pfs_design(design_id)
        else:
            tile_id = int(cmd.cmd.keywords['tile_id'].values[0])
        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        guide = False
        if 'guide' in cmd.cmd.keywords:
            guide = bool(cmd.cmd.keywords['guide'].values[0])

        try:
            cmd.inform('exposureTime={}'.format(exposure_time))
            # start an exposure
            result = self.actor.sendCommand(
                actor='agcc',
                cmdStr='expose object pfsVisitId={} exptime={} centroid=1'.format(visit_id, exposure_time / 1000),
                timeLim=(exposure_time // 1000 + 5)
            )
            telescope_state = self.actor.mlp1.telescopeState
            self.actor.logger.info('AgCmd.acquire_field: telescopeState={}'.format(telescope_state))
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info('AgCmd.acquire_field: frameId={}'.format(frame_id))
            data_time = self.actor.agcc.dataTime
            self.actor.logger.info('AgCmd.acquire_field: dataTime={}'.format(data_time))
            # retrieve field center coordinates from opdb
            # retrieve exposure information from opdb
            # retrieve guide star coordinates from opdb
            # retrieve metrics of detected objects from opdb
            # compute offsets, scale, transparency, and seeing
            if guide:
                cmd.inform('detectionState=1')
                # convert equatorial coordinates to horizontal coordinates
                dra, ddec, dinr, dalt, daz = field_acquisition.acquire_field(tile_id, frame_id, altazimuth=True, logger=self.actor.logger)
                cmd.inform('text="dra={},ddec={},dinr={},dalt={},daz={}"'.format(dra, ddec, dinr, dalt, daz))
                cmd.inform('detectionState=0')
                # send corrections to mlp1 and gen2 (or iic)
                result = self.actor.sendCommand(
                    actor='mlp1',
                    cmdStr='guide azel={},{} ready=1 time={} delay=0 xy=0,0 size=0 intensity=0 flux=0'.format(daz, dalt, data_time),
                    timeLim=5
                )
                #cmd.inform('guideReady=1')
            else:
                cmd.inform('detectionState=1')
                dra, ddec, dinr = field_acquisition.acquire_field(tile_id, frame_id, logger=self.actor.logger)
                cmd.inform('text="dra={},ddec={},dinr={}"'.format(dra, ddec, dinr))
                cmd.inform('detectionState=0')
                # send corrections to gen2 (or iic)
            # store results in opdb
        except Exception as e:
            cmd.fail('text="AgCmd.acquire_field: {}'.format(e))
            return
        cmd.finish()

    def focus(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))
        mode = controller.get_mode()
        if mode != controller.Mode.OFF:
            cmd.fail('text="AgCmd.acquire_field: mode={}'.format(mode))
            return

        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100

        try:
            cmd.inform('exposureTime={}'.format(exposure_time))
            # start an exposure
            result = self.actor.sendCommand(
                actor='agcc',
                cmdStr='expose object pfsVisitId={} exptime={} centroid=1'.format(visit_id, exposure_time / 1000),
                timeLim=(exposure_time // 1000 + 5)
            )
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info('AgCmd.focus: frameId={}'.format(frame_id))
            # retrieve detected objects from agcc (or opdb)
            # compute focus offset and tilt
            dz = focus.focus(frame_id, verbose=True, logger=self.actor.logger)
            if numpy.isnan(dz):
                cmd.fail('text="AgCmd.focus: dz={}'.format(dz))
                return
            cmd.inform('text="dz={}"'.format(dz))
            # send corrections to gen2 (or iic)
            # store results in opdb
        except Exception as e:
            cmd.fail('text="AgCmd.focus: {}'.format(e))
            return
        cmd.finish()

    def start_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        tile_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0])
            # get tile_id by design_id
            tile_id, *_ = opdb.query_pfs_design(design_id)
        elif 'tile_id' in cmd.cmd.keywords:
            tile_id = int(cmd.cmd.keywords['tile_id'].values[0])
        visit_id = None
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        from_sky = None
        if 'from_sky' in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords['from_sky'].values[0])
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = 0 # ms
        if 'cadence' in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords['cadence'].values[0])
            if cadence < 0:
                cadence = 0
        focus = False
        if 'focus' in cmd.cmd.keywords:
            focus = bool(cmd.cmd.keywords['focus'].values[0])

        try:
            controller.start_autoguide(cmd=cmd, tile_id=tile_id, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.start_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0])
            # get tile_id by design_id
            tile_id, *_ = opdb.query_pfs_design(design_id)
        else:
            tile_id = int(cmd.cmd.keywords['tile_id'].values[0])
        visit_id = self.visit_id
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            self.visit_id = visit_id
        from_sky = None
        if 'from_sky' in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords['from_sky'].values[0])
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100

        try:
            controller.initialize_autoguide(cmd=cmd, tile_id=tile_id, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time)
        except Exception as e:
            cmd.fail('text="AgCmd.initialize_autoguide: {}"'.format(e))
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
