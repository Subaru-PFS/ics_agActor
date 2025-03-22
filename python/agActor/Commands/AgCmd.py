#!/usr/bin/env python

import time

import numpy as np
import opscore.protocols.keys as keys
import opscore.protocols.types as types

from agActor import data_utils, field_acquisition, focus as _focus, pfs_design
from agActor.Controllers.ag import ag
from agActor.kawanomoto import Subaru_POPT2_PFS  # *NOT* 'from agActor.kawanomoto import Subaru_POPT2_PFS'
from agActor.telescope_center import telCenter as tel_center
from agActor.utils import FILENAMES, save_shm_files


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.vocab = [
            ('ping', '', self.ping),
            ('status', '', self.status),
            ('show', '', self.show),
            ('acquire_field',
             '[<design_id>] [<design_path>] [<visit_id>|<visit>] [<exposure_time>] [<guide>] [<offset>] [@coarse] ['
             '<dinr>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] ['
             '<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]',
             self.acquire_field),
            ('acquire_field',
             '@otf [<visit_id>|<visit>] [<exposure_time>] [<guide>] [<center>] [@coarse] [<offset>] [<dinr>] ['
             '<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] ['
             '<max_residual>] [<exposure_delay>] [<tec_off>]',
             self.acquire_field),
            ('focus',
             '[<visit_id>|<visit>] [<exposure_time>] [<max_ellipticity>] [<max_size>] [<min_size>] [<exposure_delay>] '
             '[<tec_off>]',
             self.focus),
            ('autoguide',
             '@start [<design_id>] [<design_path>] [<visit_id>|<visit>] [<from_sky>] [<exposure_time>] [<cadence>] ['
             '<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] ['
             '<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]',
             self.start_autoguide),
            ('autoguide',
             '@start @otf [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] [<dry_run>] ['
             '<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] ['
             '<exposure_delay>] [<tec_off>]',
             self.start_autoguide),
            ('autoguide',
             '@initialize [<design_id>] [<design_path>] [<visit_id>|<visit>] [<from_sky>] [<exposure_time>] ['
             '<cadence>] [<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] ['
             '<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]',
             self.initialize_autoguide),
            ('autoguide',
             '@initialize @otf [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] ['
             '<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] ['
             '<exposure_delay>] [<tec_off>]',
             self.initialize_autoguide),
            ('autoguide', '@restart', self.restart_autoguide),
            ('autoguide', '@stop', self.stop_autoguide),
            ('autoguide',
             '@reconfigure [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] '
             '[<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]',
             self.reconfigure_autoguide),
            ('offset', '[@(absolute|relative)] [<dx>] [<dy>] [<dinr>] [<dscale>]', self.offset),
            ('offset', '@reset', self.offset),
        ]
        self.keys = keys.KeysDictionary(
            'ag_ag',
            (1, 15),
            keys.Key('exposure_time', types.Int(), help=''),
            keys.Key('cadence', types.Int(), help=''),
            keys.Key('guide', types.Bool('no', 'yes'), help=''),
            keys.Key('design_id', types.String(), help=''),
            keys.Key('design_path', types.String(), help=''),
            keys.Key('initial', types.Bool('no', 'yes'), help=''),
            keys.Key('visit_id', types.Int(), help=''),
            keys.Key('visit', types.Int(), help=''),
            keys.Key('from_sky', types.Bool('no', 'yes'), help=''),
            keys.Key('center', types.Float() * (2, 3), help=''),
            keys.Key('offset', types.Float() * (2, 4), help=''),
            keys.Key('magnitude', types.Float(), help=''),
            keys.Key('dry_run', types.Bool('no', 'yes'), help=''),
            keys.Key('dx', types.Float(), help=''),
            keys.Key('dy', types.Float(), help=''),
            keys.Key('dinr', types.Float(), help=''),
            keys.Key('dscale', types.Float(), help=''),
            keys.Key('fit_dinr', types.Bool('no', 'yes'), help=''),
            keys.Key('fit_dscale', types.Bool('no', 'yes'), help=''),
            keys.Key('max_ellipticity', types.Float(), help=''),
            keys.Key('max_size', types.Float(), help=''),
            keys.Key('min_size', types.Float(), help=''),
            keys.Key('max_residual', types.Float(), help=''),
            keys.Key('exposure_delay', types.Int(), help=''),
            keys.Key('tec_off', types.Bool('no', 'yes'), help=''),
        )
        self.with_opdb_agc_guide_offset = actor.actorConfig.get('agc_guide_offset', False)
        self.with_opdb_agc_match = actor.actorConfig.get('agc_match', False)
        self.with_agcc_timestamp = actor.actorConfig.get('agcc_timestamp', False)

        tel_status = [x.strip() for x in actor.actorConfig.get('tel_status', ('agc_exposure',)).split(',')]
        self.with_gen2_status = 'gen2' in tel_status
        self.with_mlp1_status = 'mlp1' in tel_status
        self.with_opdb_tel_status = 'tel_status' in tel_status
        self.with_design_path = actor.actorConfig.get('design_path', '').strip() or None

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
                self.actor.logger.exception('AgCmd.show:')
                cmd.warn('text="AgCmd.show: {}: {}"'.format(n, e))
        cmd.finish()

    def acquire_field(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))
        mode = controller.get_mode()

        # Don't run this command if the controller is already in use.
        if mode != controller.Mode.OFF:
            cmd.fail('text="AgCmd.acquire_field: mode={}"'.format(mode))
            return

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
        elif 'visit' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit'].values[0])
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        guide = True
        if 'guide' in cmd.cmd.keywords:
            guide = bool(cmd.cmd.keywords['guide'].values[0])
        center = None
        if 'center' in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords['center'].values])
        offset = None
        if 'offset' in cmd.cmd.keywords:
            offset = tuple([float(x) for x in cmd.cmd.keywords['offset'].values])

        dinr = None
        if 'dinr' in cmd.cmd.keywords:
            dinr = float(cmd.cmd.keywords['dinr'].values[0])

        kwargs = {}
        kwargs['coarse'] = ag.COARSE
        if 'coarse' in cmd.cmd.keywords:
            kwargs['coarse'] = bool(cmd.cmd.keywords['coarse'].values[0])
        if 'magnitude' in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords['magnitude'].values[0])
            kwargs['magnitude'] = magnitude
        dry_run = ag.DRY_RUN
        if 'dry_run' in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords['dry_run'].values[0])
        if 'fit_dinr' in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords['fit_dinr'].values[0])
            kwargs['fit_dinr'] = fit_dinr
        if 'fit_dscale' in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords['fit_dscale'].values[0])
            kwargs['fit_dscale'] = fit_dscale
        if 'max_ellipticity' in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords['max_ellipticity'].values[0])
            kwargs['max_ellipticity'] = max_ellipticity
        if 'max_size' in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords['max_size'].values[0])
            kwargs['max_size'] = max_size
        if 'min_size' in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords['min_size'].values[0])
            kwargs['min_size'] = min_size
        if 'max_residual' in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords['max_residual'].values[0])
            kwargs['max_residual'] = max_residual
        exposure_delay = ag.EXPOSURE_DELAY
        if 'exposure_delay' in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords['exposure_delay'].values[0])
        tec_off = ag.TEC_OFF
        if 'tec_off' in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords['tec_off'].values[0])

        try:
            cmd.inform(f"exposureTime={exposure_time}")

            # start an exposure
            cmdStr = f"expose object exptime={exposure_time / 1000} centroid=1"
            if visit_id is not None:
                cmdStr += f' visit={visit_id}'
            if exposure_delay > 0:
                cmdStr += f' threadDelay={exposure_delay}'
            if tec_off:
                cmdStr += ' tecOFF'

            result = self.actor.queueCommand(
                actor='agcc',
                cmdStr=cmdStr,
                timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 5)
            )
            time.sleep((exposure_time + 7 * exposure_delay) / 1000 / 2)
            telescope_state = None

            if self.with_mlp1_status:
                telescope_state = self.actor.mlp1.telescopeState
                self.actor.logger.info(f"AgCmd.acquire_field: telescopeState={telescope_state}")
                kwargs['inr'] = telescope_state['rotator_real_angle']

            if self.with_gen2_status or self.with_opdb_tel_status:
                # update gen2 status values
                self.actor.queueCommand(
                    actor='gen2',
                    cmdStr=f"updateTelStatus caller={self.actor.name}",
                    timeLim=5
                ).get()

                if self.with_gen2_status:
                    tel_status = self.actor.gen2.tel_status
                    self.actor.logger.info(f"AgCmd.acquire_field: tel_status={tel_status}")
                    kwargs['tel_status'] = tel_status
                    _tel_center = tel_center(actor=self.actor, center=center, design=design, tel_status=tel_status)
                    if all(x is None for x in (center, design)):
                        center, _offset = _tel_center.dither  # dithered center and guide offset correction (insrot
                        # only)
                        self.actor.logger.info(f"AgCmd.acquire_field: center={center}")
                    else:
                        _offset = _tel_center.offset  # dithering and guide offset correction
                    if offset is None:
                        offset = _offset
                        self.actor.logger.info(f"AgCmd.acquire_field: offset={offset}")

                if self.with_opdb_tel_status:
                    status_update = self.actor.gen2.statusUpdate
                    status_id = (status_update['visit'], status_update['sequenceNum'])
                    self.actor.logger.info(f"AgCmd.acquire_field: status_id={status_id}")
                    kwargs['status_id'] = status_id

            # wait for an exposure to complete
            result.get()

            # Do things with the exposure.
            frame_id = self.actor.agcc.frameId
            data_time = self.actor.agcc.dataTime
            taken_at = data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2

            self.actor.logger.info(f"AgCmd.acquire_field: frameId={frame_id}")
            self.actor.logger.info(f"AgCmd.acquire_field: dataTime={data_time}")
            self.actor.logger.info(f"AgCmd.acquire_field: taken_at={taken_at}")

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

            if dinr is not None:
                kwargs['dinr'] = dinr

            # retrieve field center coordinates from opdb
            # retrieve exposure information from opdb
            # retrieve guide star coordinates from opdb
            # retrieve metrics of detected objects from opdb
            # compute offsets, scale, transparency, and seeing

            cmd.inform('detectionState=1')
            offset_info = field_acquisition.acquire_field(
                design=design,
                frame_id=frame_id,
                altazimuth=guide,
                logger=self.actor.logger,
                **kwargs
            )

            # If there are no detected objects, fail the command.
            if len(offset_info.detected_objects) == 0:
                cmd.fail('text="AgCmd.acquire_field: No detected objects found, can\'t compute offset"')
                return

            # TODO: remove these and use offset_info directly.
            ra = offset_info.ra
            dec = offset_info.dec
            inst_pa = offset_info.inst_pa
            dx = offset_info.dx
            dy = offset_info.dy
            dra = offset_info.dra
            ddec = offset_info.ddec
            dinr = offset_info.dinr
            dscale = offset_info.dscale
            dalt = offset_info.dalt
            daz = offset_info.daz
            spot_size = offset_info.spot_size
            peak_intensity = offset_info.peak_intensity
            flux = offset_info.flux

            # Save the detected, guide, and identified objects.
            save_shm_files(offset_info)

            if guide:
                cmd.inform(f'text="{ra=},{dec=},{inst_pa=},{dra=},{ddec=},{dinr=},{dscale=},{dalt=},{daz=}"')
            else:
                cmd.inform(f'text="dra={dra},ddec={ddec},dinr={dinr},dscale={dscale}"')
            cmd.inform(
                'data={},{},{},"{}","{}","{}"'.format(offset_info.ra, offset_info.dec, offset_info.inst_pa, *FILENAMES)
                )
            cmd.inform('detectionState=0')

            if guide:
                # send corrections to mlp1 and gen2 (or iic)
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
                result.get()
                # cmd.inform('guideReady=1')

            # always compute focus offset and tilt
            kwargs = {
                key: kwargs.get(key)
                for key in ('max_ellipticity', 'max_size', 'min_size')
                if key in kwargs
            }
            dz, dzs = _focus._focus(
                detected_objects=offset_info.detected_objects, logger=self.actor.logger
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
                    identified_objects=offset_info.identified_objects,
                )
        except Exception as e:
            self.actor.logger.exception('AgCmd.acquire_field:')
            cmd.fail(f'text="AgCmd.acquire_field: {e}"')
            return
        cmd.finish()

    def focus(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))
        mode = controller.get_mode()
        if mode != controller.Mode.OFF:
            cmd.fail('text="AgCmd.focus: mode={}"'.format(mode))
            return

        visit_id = None
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
        elif 'visit' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit'].values[0])
        exposure_time = 2000  # ms
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
        kwargs = {}
        if 'max_ellipticity' in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords['max_ellipticity'].values[0])
            kwargs['max_ellipticity'] = max_ellipticity
        if 'max_size' in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords['max_size'].values[0])
            kwargs['max_size'] = max_size
        if 'min_size' in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords['min_size'].values[0])
            kwargs['min_size'] = min_size
        exposure_delay = ag.EXPOSURE_DELAY
        if 'exposure_delay' in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords['exposure_delay'].values[0])
        tec_off = ag.TEC_OFF
        if 'tec_off' in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords['tec_off'].values[0])

        try:
            cmd.inform('exposureTime={}'.format(exposure_time))
            # start an exposure
            cmdStr = 'expose object exptime={} centroid=1'.format(exposure_time / 1000)
            if visit_id is not None:
                cmdStr += ' visit={}'.format(visit_id)
            if exposure_delay > 0:
                cmdStr += ' threadDelay={}'.format(exposure_delay)
            if tec_off:
                cmdStr += ' tecOFF'
            result = self.actor.queueCommand(
                actor='agcc',
                cmdStr=cmdStr,
                timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 5)
            )
            # wait for an exposure to complete
            result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info('AgCmd.focus: frameId={}'.format(frame_id))
            # retrieve detected objects from agcc (or opdb)
            # compute focus offset and tilt
            dz, dzs = _focus.focus(frame_id=frame_id, logger=self.actor.logger, **kwargs)
            if np.isnan(dz):
                cmd.fail('text="AgCmd.focus: dz={}"'.format(dz))
                return
            cmd.inform('text="dz={}"'.format(dz))
            # send corrections to gen2 (or iic)
            cmd.inform(
                'guideErrors={},{},{},{},{},{},{},{}'.format(
                    frame_id, np.nan, np.nan, np.nan, np.nan, np.nan, dz, np.nan
                    )
                )
            cmd.inform('focusErrors={},{},{},{},{},{},{}'.format(frame_id, *dzs))
            # store results in opdb
            if self.with_opdb_agc_guide_offset:
                data_utils.write_agc_guide_offset(
                    frame_id=frame_id,
                    delta_z=dz,
                    delta_zs=dzs
                )
        except Exception as e:
            self.actor.logger.exception('AgCmd.focus:')
            cmd.fail('text="AgCmd.focus: {}"'.format(e))
            return
        cmd.finish()

    def start_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
        elif 'visit' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit'].values[0])
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
        center = None
        if 'center' in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords['center'].values])

        kwargs = {}
        coarse = ag.COARSE
        if 'coarse' in cmd.cmd.keywords:
            coarse = bool(cmd.cmd.keywords['coarse'].values[0])
            kwargs['coarse'] = coarse
        if 'magnitude' in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords['magnitude'].values[0])
            kwargs['magnitude'] = magnitude
        if 'dry_run' in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords['dry_run'].values[0])
            kwargs['dry_run'] = dry_run
        if 'fit_dinr' in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords['fit_dinr'].values[0])
            kwargs['fit_dinr'] = fit_dinr
        if 'fit_dscale' in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords['fit_dscale'].values[0])
            kwargs['fit_dscale'] = fit_dscale
        if 'max_ellipticity' in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords['max_ellipticity'].values[0])
            kwargs['max_ellipticity'] = max_ellipticity
        if 'max_size' in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords['max_size'].values[0])
            kwargs['max_size'] = max_size
        if 'min_size' in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords['min_size'].values[0])
            kwargs['min_size'] = min_size
        if 'max_residual' in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords['max_residual'].values[0])
            kwargs['max_residual'] = max_residual
        if 'exposure_delay' in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords['exposure_delay'].values[0])
            kwargs['exposure_delay'] = exposure_delay
        if 'tec_off' in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords['tec_off'].values[0])
            kwargs['tec_off'] = tec_off

        try:
            controller.start_autoguide(
                cmd=cmd, design=design, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time,
                cadence=cadence, center=center, **kwargs
                )
        except Exception as e:
            self.actor.logger.exception('AgCmd.start_autoguide:')
            cmd.fail('text="AgCmd.start_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))

        design_id = None
        if 'design_id' in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords['design_id'].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if 'design_path' in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords['design_path'].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
        elif 'visit' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit'].values[0])
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
        center = None
        if 'center' in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords['center'].values])

        kwargs = {}
        coarse = ag.COARSE
        if 'coarse' in cmd.cmd.keywords:
            coarse = bool(cmd.cmd.keywords['coarse'].values[0])
            kwargs['coarse'] = coarse
        if 'magnitude' in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords['magnitude'].values[0])
            kwargs['magnitude'] = magnitude
        if 'dry_run' in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords['dry_run'].values[0])
            kwargs['dry_run'] = dry_run
        if 'fit_dinr' in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords['fit_dinr'].values[0])
            kwargs['fit_dinr'] = fit_dinr
        if 'fit_dscale' in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords['fit_dscale'].values[0])
            kwargs['fit_dscale'] = fit_dscale
        if 'max_ellipticity' in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords['max_ellipticity'].values[0])
            kwargs['max_ellipticity'] = max_ellipticity
        if 'max_size' in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords['max_size'].values[0])
            kwargs['max_size'] = max_size
        if 'min_size' in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords['min_size'].values[0])
            kwargs['min_size'] = min_size
        if 'max_residual' in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords['max_residual'].values[0])
            kwargs['max_residual'] = max_residual
        if 'exposure_delay' in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords['exposure_delay'].values[0])
            kwargs['exposure_delay'] = exposure_delay
        if 'tec_off' in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords['tec_off'].values[0])
            kwargs['tec_off'] = tec_off

        try:
            controller.initialize_autoguide(
                cmd=cmd, design=design, visit_id=visit_id, from_sky=from_sky, exposure_time=exposure_time,
                cadence=cadence, center=center, **kwargs
                )
        except Exception as e:
            self.actor.logger.exception('AgCmd.initialize_autoguide:')
            cmd.fail('text="AgCmd.initialize_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def restart_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))

        try:
            controller.restart_autoguide(cmd=cmd)
        except Exception as e:
            self.actor.logger.exception('AgCmd.restart_autoguide:')
            cmd.fail('text="AgCmd.restart_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def stop_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))

        try:
            controller.stop_autoguide()
        except Exception as e:
            self.actor.logger.exception('AgCmd.stop_autoguide:')
            cmd.fail('text="AgCmd.stop_autoguide: {}"'.format(e))
            return
        cmd.finish()

    def reconfigure_autoguide(self, cmd):

        controller = self.actor.controllers['ag']
        # self.actor.logger.info('controller={}'.format(controller))

        kwargs = {}
        if 'visit_id' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit_id'].values[0])
            kwargs['visit_id'] = visit_id
        elif 'visit' in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords['visit'].values[0])
            kwargs['visit_id'] = visit_id
        if 'exposure_time' in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords['exposure_time'].values[0])
            if exposure_time < 100:
                exposure_time = 100
            kwargs['exposure_time'] = exposure_time
        if 'cadence' in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords['cadence'].values[0])
            if cadence < 0:
                cadence = 0
            kwargs['cadence'] = cadence
        if 'dry_run' in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords['dry_run'].values[0])
            kwargs['dry_run'] = dry_run
        if 'fit_dinr' in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords['fit_dinr'].values[0])
            kwargs['fit_dinr'] = fit_dinr
        if 'fit_dscale' in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords['fit_dscale'].values[0])
            kwargs['fit_dscale'] = fit_dscale
        if 'max_ellipticity' in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords['max_ellipticity'].values[0])
            kwargs['max_ellipticity'] = max_ellipticity
        if 'max_size' in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords['max_size'].values[0])
            kwargs['max_size'] = max_size
        if 'min_size' in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords['min_size'].values[0])
            kwargs['min_size'] = min_size
        if 'max_residual' in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords['max_residual'].values[0])
            kwargs['max_residual'] = max_residual
        if 'exposure_delay' in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords['exposure_delay'].values[0])
            kwargs['exposure_delay'] = exposure_delay
        if 'tec_off' in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords['tec_off'].values[0])
            kwargs['tec_off'] = tec_off

        try:
            controller.reconfigure_autoguide(cmd=cmd, **kwargs)
        except Exception as e:
            self.actor.logger.exception('AgCmd.reconfigure_autoguide:')
            cmd.fail('text="AgCmd.reconfigure_autoguide: {}"'.format(e))
            return
        cmd.finish()

    _SCALE0 = Subaru_POPT2_PFS.Unknown_Scale_Factor_AG

    def offset(self, cmd):

        # controller = self.actor.controllers['ag']
        # #self.actor.logger.info('controller={}'.format(controller))
        # mode = controller.get_mode()
        # if mode != controller.Mode.OFF:
        #     cmd.fail('text="AgCmd.offset: mode={}"'.format(mode))
        #     return

        def zero_offset(*, dx=None, dy=None, dinr=None, dscale=None, relative=False):

            if dx is not None:
                if relative:
                    Subaru_POPT2_PFS.inr_axis_on_dp_x += dx
                else:
                    Subaru_POPT2_PFS.inr_axis_on_dp_x = dx
            if dy is not None:
                if relative:
                    Subaru_POPT2_PFS.inr_axis_on_dp_y += dy
                else:
                    Subaru_POPT2_PFS.inr_axis_on_dp_y = dy
            if dinr is not None:
                if relative:
                    Subaru_POPT2_PFS.inr_zero_offset += dinr
                else:
                    Subaru_POPT2_PFS.inr_zero_offset = dinr
            if dscale is not None:
                if relative:
                    Subaru_POPT2_PFS.Unknown_Scale_Factor_AG += dscale
                else:
                    Subaru_POPT2_PFS.Unknown_Scale_Factor_AG = AgCmd._SCALE0 + dscale
            return (Subaru_POPT2_PFS.inr_axis_on_dp_x, Subaru_POPT2_PFS.inr_axis_on_dp_y,
                    Subaru_POPT2_PFS.inr_zero_offset, Subaru_POPT2_PFS.Unknown_Scale_Factor_AG - AgCmd._SCALE0)

        dx = None
        if 'dx' in cmd.cmd.keywords:
            dx = float(cmd.cmd.keywords['dx'].values[0])
        dy = None
        if 'dy' in cmd.cmd.keywords:
            dy = float(cmd.cmd.keywords['dy'].values[0])
        dinr = None
        if 'dinr' in cmd.cmd.keywords:
            dinr = float(cmd.cmd.keywords['dinr'].values[0])
        dscale = None
        if 'dscale' in cmd.cmd.keywords:
            dscale = float(cmd.cmd.keywords['dscale'].values[0])
        if 'reset' in cmd.cmd.keywords:
            dx, dy, dinr, dscale = 0.0, 0.0, -90.0, 0.0
        dx, dy, dinr, dscale = zero_offset(
            dx=dx, dy=dy, dinr=dinr, dscale=dscale, relative='relative' in cmd.cmd.keywords
            )
        self.actor.logger.info('AgCmd.offset: dx={},dy={},dinr={},dscale={}'.format(dx, dy, dinr, dscale))
        cmd.inform('text="dx={},dy={},dinr={},dscale={}"'.format(dx, dy, dinr, dscale))
        cmd.inform('guideOffsets={},{}'.format(dx, dy))
        cmd.finish()
