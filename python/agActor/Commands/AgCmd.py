#!/usr/bin/env python

from twisted.internet import reactor
import opscore.protocols.keys as keys
import opscore.protocols.types as types


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('show', '', self.show),
            ('acquire_field', '<target_id> [<exposure_time>]', self.acquire_field),
            ('focus', '[<exposure_time>]', self.focus),
            ('autoguide', 'start [<initialize>] [<exposure_time>] [<cadence>] [<focus>]', self.start_autoguide),
            ('autoguide', 'initialize [<exposure_time>]', self.initialize_autoguide),
            ('autoguide', 'stop', self.stop_autoguide),
        ]
        self.keys = keys.KeysDictionary(
            'ag_ag',
            (1, 1),
            keys.Key('initialize', types.Enum('no', 'yes'), help=''),
            keys.Key('exposure_time', types.Int(), help=''),
            keys.Key('cadence', types.Int(), help=''),
            keys.Key('focus', types.Enum('no', 'yes'), help=''),
            keys.Key('target_id', types.Int(), help=''),
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
        # start an exposure
        try:
            result = self.actor.sendCommand(
                actor='agcam',
                cmdStr='expose speed={}'.format(exposure_time),
                timeLim=(exposure_time // 1000 + 5)
            )
        except Exception as e:
            cmd.fail('text="AgCmd.acquire_field: {}'.format(e))
            return
        # get a list of detected objects from agcam
        # get field center coordinates from opDB
        # get a list of guide stars from opDB
        # compute offsets, scale, transparency, and seeing
        # write results to opDB
        # send corrections to mlp1 and gen2 (or iic)
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
        # start an exposure
        try:
            result = self.actor.sendCommand(
                actor='agcam',
                cmdStr='expose speed={}'.format(exposure_time),
                timeLim=(exposure_time // 1000 + 5)
            )
        except Exception as e:
            cmd.fail('text="AgCmd.focus: {}'.format(e))
            return
        # get a list of detected objects from agcam
        # compute focus offset and tilt
        # write results to opDB
        # send corrections to gen2 (or iic)
        cmd.finish()

    def start_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        try:
            initialize = True
            if 'initialize' in cmd.cmd.keywords:
                initialize = bool(cmd.cmd.keywords['initialize'].values[0])
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
            focus = True if 'focus' in cmd.cmd.keywords and cmd.cmd.keywords['focus'].values[0] == 'yes' else False
            controller.start_autoguide(cmd=cmd, initialize=initialize, exposure_time=exposure_time, cadence=cadence, focus=focus)
        except Exception as e:
            cmd.fail('text="AgCmd.start_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        #self.actor.logger.info('controller={}'.format(controller))

        try:
            exposure_time = 2000 # ms
            if 'exposure_time' in cmd.cmd.keywords:
                exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
                if exposure_time < 100:
                    exposure_time = 100
            controller.initialize_autoguide(cmd=cmd, exposure_time=exposure_time)
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
