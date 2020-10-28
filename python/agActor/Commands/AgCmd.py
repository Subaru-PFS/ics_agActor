#!/usr/bin/env python

import opscore.protocols.keys as keys
import opscore.protocols.types as types
from agActor import field_acquisition


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('show', '', self.show),
            ('acquire_field', '<target_id> [<exposure_time>] [<guide>]', self.acquire_field),
            ('focus', '[<exposure_time>]', self.focus),
            ('autoguide', 'start [<target_id>] [<from_sky>] [<exposure_time>] [<cadence>] [<focus>]', self.start_autoguide),
            ('autoguide', 'initialize <target_id> [<from_sky>] [<exposure_time>]', self.initialize_autoguide),
            ('autoguide', 'stop', self.stop_autoguide),
            ('autoguide', 'reconfigure [<exposure_time>] [<cadence>] [<focus>]', self.reconfigure_autoguide),
        ]
        self.keys = keys.KeysDictionary(
            'ag_ag',
            (1, 3),
            keys.Key('exposure_time', types.Int(), help=''),
            keys.Key('cadence', types.Int(), help=''),
            keys.Key('focus', types.Enum('no', 'yes'), help=''),
            keys.Key('guide', types.Enum('no', 'yes'), help=''),
            keys.Key('target_id', types.Int(), help=''),
            keys.Key('from_sky', types.Enum('no', 'yes'), help=''),
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

        target_id = int(cmd.cmd.keywords['target_id'].values[0])
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        guide = True
        if 'guide' in cmd.cmd.keywords:
            guide = bool(cmd.cmd.keywords['guide'].values[0])

        try:
            # start an exposure
            cmdStr = 'expose speed={}'.format(exposure_time)
            result = self.actor.sendCommand(
                actor='agcam',
                cmdStr=cmdStr,
                timeLim=(exposure_time // 1000 + 5)
            )
            telescope_state = [
                t(v) for t, v in zip(
                    (int, int, float, float, float, float, int, int),
                    self.actor.models['mlp1'].keyVarDict['telescopeState'].valueList
                )
            ]
            self.actor.logger.info('AgCmd.acquire_field: telescopeState={}'.format(telescope_state))
            frame_id = int(self.actor.models['agcam'].keyVarDict['frameId'].valueList[0])
            self.actor.logger.info('AgCmd.acquire_field: frameId={}'.format(frame_id))
            # retrieve field center coordinates from opdb
            # retrieve exposure information from opdb
            # retrieve guide star coordinates from opdb
            # retrieve metrics of detected objects from opdb
            # compute offsets, scale, transparency, and seeing
            cmd.inform('detectionState=1')
            dra, ddec, dinr = field_acquisition.acquire_field(target_id, frame_id, logger=self.actor.logger)
            cmd.inform('text="dra={},ddec={},dinr={}"'.format(dra, ddec, dinr))
            cmd.inform('detectionState=0')
            # store results in opdb
            if guide:
                # convert equatorial coordinates to horizontal coordinates
                # send corrections to mlp1 and gen2 (or iic)
                pass
            else:
                # send corrections to gen2 (or iic)
                pass
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

        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100

        try:
            # start an exposure
            result = self.actor.sendCommand(
                actor='agcam',
                cmdStr='expose speed={}'.format(exposure_time),
                timeLim=(exposure_time // 1000 + 5)
            )
            # retrieve detected objects from agcam (or opdb)
            # compute focus offset and tilt
            # store results in opdb
            # send corrections to gen2 (or iic)
        except Exception as e:
            cmd.fail('text="AgCmd.focus: {}'.format(e))
            return
        cmd.finish()

    def start_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        target_id = None
        if 'target_id' in cmd.cmd.keywords:
            target_id = int(cmd.cmd.keywords['target_id'].values[0])
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
            controller.start_autoguide(cmd=cmd, target_id=target_id, from_sky=from_sky, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.start_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        target_id = int(cmd.cmd.keywords['target_id'].values[0])
        from_sky = None
        if 'from_sky' in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords['from_sky'].values[0])
        exposure_time = 2000 # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100

        try:
            controller.initialize_autoguide(cmd=cmd, target_id=target_id, from_sky=from_sky, exposure_time=exposure_time)
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
