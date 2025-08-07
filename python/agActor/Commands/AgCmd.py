#!/usr/bin/env python

import time

import numpy as np
import opscore.protocols.keys as keys
import opscore.protocols.types as types
from pfs.utils.coordinates import Subaru_POPT2_PFS

from agActor import field_acquisition
from agActor.catalog import pfs_design
from agActor.Controllers.ag import ag
from agActor.utils import actorCalls
from agActor.utils import data as data_utils
from agActor.utils import focus as _focus
from agActor.utils.telescope_center import telCenter as tel_center


class AgCmd:

    def __init__(self, actor):

        self.actor = actor
        self.vocab = [
            ("ping", "", self.ping),
            ("status", "", self.status),
            ("show", "", self.show),
            (
                "acquire_field",
                "[<design_id>] [<design_path>] [<visit_id>|<visit>] [<exposure_time>] [<guide>] [<offset>] [<dinr>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.acquire_field,
            ),
            (
                "acquire_field",
                "@otf [<visit_id>|<visit>] [<exposure_time>] [<guide>] [<center>] [<offset>] [<dinr>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.acquire_field,
            ),
            (
                "focus",
                "[<visit_id>|<visit>] [<exposure_time>] [<max_ellipticity>] [<max_size>] [<min_size>] [<exposure_delay>] [<tec_off>]",
                self.focus,
            ),
            (
                "autoguide",
                "@start [<design_id>] [<design_path>] [<visit_id>|<visit>] [<from_sky>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.start_autoguide,
            ),
            (
                "autoguide",
                "@start @otf [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.start_autoguide,
            ),
            (
                "autoguide",
                "@initialize [<design_id>] [<design_path>] [<visit_id>|<visit>] [<from_sky>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.initialize_autoguide,
            ),
            (
                "autoguide",
                "@initialize @otf [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<center>] [<magnitude>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.initialize_autoguide,
            ),
            ("autoguide", "@restart", self.restart_autoguide),
            ("autoguide", "@stop", self.stop_autoguide),
            (
                "autoguide",
                "@reconfigure [<visit_id>|<visit>] [<exposure_time>] [<cadence>] [<dry_run>] [<fit_dinr>] [<fit_dscale>] [<max_ellipticity>] [<max_size>] [<min_size>] [<max_residual>] [<exposure_delay>] [<tec_off>]",
                self.reconfigure_autoguide,
            ),
            ("offset", "[@(absolute|relative)] [<dx>] [<dy>] [<dinr>] [<dscale>]", self.offset),
            ("offset", "@reset", self.offset),
        ]
        self.keys = keys.KeysDictionary(
            "ag_ag",
            (1, 15),
            keys.Key("exposure_time", types.Int(), help=""),
            keys.Key("cadence", types.Int(), help=""),
            keys.Key("guide", types.Bool("no", "yes"), help=""),
            keys.Key("design_id", types.String(), help=""),
            keys.Key("design_path", types.String(), help=""),
            keys.Key("visit_id", types.Int(), help=""),
            keys.Key("visit", types.Int(), help=""),
            keys.Key("from_sky", types.Bool("no", "yes"), help=""),
            keys.Key("center", types.Float() * (2, 3), help=""),
            keys.Key("offset", types.Float() * (2, 4), help=""),
            keys.Key("magnitude", types.Float(), help=""),
            keys.Key("dry_run", types.Bool("no", "yes"), help=""),
            keys.Key("dx", types.Float(), help=""),
            keys.Key("dy", types.Float(), help=""),
            keys.Key("dinr", types.Float(), help=""),
            keys.Key("dscale", types.Float(), help=""),
            keys.Key("fit_dinr", types.Bool("no", "yes"), help=""),
            keys.Key("fit_dscale", types.Bool("no", "yes"), help=""),
            keys.Key("max_ellipticity", types.Float(), help=""),
            keys.Key("max_size", types.Float(), help=""),
            keys.Key("min_size", types.Float(), help=""),
            keys.Key("max_residual", types.Float(), help=""),
            keys.Key("exposure_delay", types.Int(), help=""),
            keys.Key("tec_off", types.Bool("no", "yes"), help=""),
        )
        self.with_opdb_agc_guide_offset = actor.actorConfig.get("agc_guide_offset", False)
        self.with_opdb_agc_match = actor.actorConfig.get("agc_match", False)
        self.with_agcc_timestamp = actor.actorConfig.get("agcc_timestamp", False)

        tel_status = [x.strip() for x in actor.actorConfig.get("tel_status", ("agc_exposure",)).split(",")]
        self.with_gen2_status = "gen2" in tel_status
        self.with_mlp1_status = "mlp1" in tel_status
        self.with_opdb_tel_status = "tel_status" in tel_status
        self.with_design_path = actor.actorConfig.get("design_path", "").strip() or None

    def ping(self, cmd):
        """Return a product name."""

        cmd.inform('text="{}"'.format(self.actor.productName))
        cmd.finish()

    def status(self, cmd):
        """Return status keywords."""

        self.actor.sendVersionKey(cmd)
        cmd.finish()

    def show(self, cmd):
        """Show status keywords from all models."""

        for n in self.actor.models:
            try:
                d = self.actor.models[n].keyVarDict
                for k, v in d.items():
                    cmd.inform('text="{}"'.format(repr(v)))
            except Exception as e:
                self.actor.logger.exception("AgCmd.show:")
                cmd.warn(f'text="AgCmd.show: {n}: {e}"')
        cmd.finish()

    def acquire_field(self, cmd):

        controller = self.actor.controllers["ag"]
        mode = controller.get_mode()
        self.actor.logger.info(f"AgCmd.acquire_field: mode={mode}")
        if mode != controller.Mode.OFF:
            cmd.fail(f'text="AgCmd.acquire_field: mode={mode}"')
            return

        design_id = None
        if "design_id" in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords["design_id"].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if "design_path" in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords["design_path"].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
        exposure_time = 2000  # ms
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
        guide = True
        if "guide" in cmd.cmd.keywords:
            guide = bool(cmd.cmd.keywords["guide"].values[0])
        center = None
        if "center" in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords["center"].values])
        offset = None
        if "offset" in cmd.cmd.keywords:
            offset = tuple([float(x) for x in cmd.cmd.keywords["offset"].values])
        dinr = None
        if "dinr" in cmd.cmd.keywords:
            dinr = float(cmd.cmd.keywords["dinr"].values[0])
        kwargs = {}
        if "magnitude" in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords["magnitude"].values[0])
            kwargs["magnitude"] = magnitude
        dry_run = ag.DRY_RUN
        if "dry_run" in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords["dry_run"].values[0])
        if "fit_dinr" in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords["fit_dinr"].values[0])
            kwargs["fit_dinr"] = fit_dinr
        if "fit_dscale" in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords["fit_dscale"].values[0])
            kwargs["fit_dscale"] = fit_dscale
        if "max_ellipticity" in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords["max_ellipticity"].values[0])
            kwargs["max_ellipticity"] = max_ellipticity
        if "max_size" in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords["max_size"].values[0])
            kwargs["max_size"] = max_size
        if "min_size" in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords["min_size"].values[0])
            kwargs["min_size"] = min_size
        if "max_residual" in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords["max_residual"].values[0])
            kwargs["max_residual"] = max_residual
        exposure_delay = ag.EXPOSURE_DELAY
        if "exposure_delay" in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords["exposure_delay"].values[0])
        tec_off = ag.TEC_OFF
        if "tec_off" in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords["tec_off"].values[0])

        self.actor.logger.info(f"AgCmd.acquire_field: kwargs={kwargs}")

        try:
            cmd.inform(f"exposureTime={exposure_time}")
            # start an exposure
            cmdStr = f"expose object exptime={exposure_time / 1000} centroid=1"
            if visit_id is not None:
                cmdStr += f" visit={visit_id}"
            if exposure_delay > 0:
                cmdStr += f" threadDelay={exposure_delay}"
            if tec_off:
                cmdStr += " tecOFF"

            self.actor.logger.info(f"AgCmd.acquire_field: Sending agcc cmdStr={cmdStr}")
            agcc_exposure_result = self.actor.queueCommand(
                actor="agcc", cmdStr=cmdStr, timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 15)
            )
            # This synchronous sleep is to defer the request for telescope info to roughly
            # the middle of the exposure.
            time.sleep((exposure_time + 7 * exposure_delay) / 1000 / 2)
            telescope_state = None
            if self.with_mlp1_status:
                telescope_state = self.actor.mlp1.telescopeState
                self.actor.logger.info(f"AgCmd.acquire_field: telescopeState={telescope_state}")
                kwargs["inr"] = telescope_state["rotator_real_angle"]
            if self.with_gen2_status or self.with_opdb_tel_status:
                if self.with_gen2_status:
                    # update gen2 status values
                    tel_status = actorCalls.updateTelStatus(self.actor, self.actor.logger, visit_id)
                    self.actor.logger.info(f"AgCmd.acquire_field: tel_status={tel_status}")
                    kwargs["tel_status"] = tel_status
                    _tel_center = tel_center(
                        actor=self.actor, center=center, design=design, tel_status=tel_status
                    )
                    if all(x is None for x in (center, design)):
                        center, _offset = (
                            _tel_center.dither
                        )  # dithered center and guide offset correction (insrot only)
                        self.actor.logger.info(f"AgCmd.acquire_field: center={center}")
                    else:
                        _offset = _tel_center.offset  # dithering and guide offset correction
                    if offset is None:
                        offset = _offset
                        self.actor.logger.info(f"AgCmd.acquire_field: offset={offset}")
                if self.with_opdb_tel_status:
                    status_update = self.actor.gen2.statusUpdate
                    status_id = (status_update["visit"], status_update["sequenceNum"])
                    self.actor.logger.info(f"AgCmd.acquire_field: status_id={status_id}")
                    kwargs["status_id"] = status_id
            # wait for an exposure to complete
            agcc_exposure_result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info(f"AgCmd.acquire_field: frameId={frame_id}")
            data_time = self.actor.agcc.dataTime
            self.actor.logger.info(f"AgCmd.acquire_field: dataTime={data_time}")
            taken_at = data_time + (exposure_time + 7 * exposure_delay) / 1000 / 2
            self.actor.logger.info(f"AgCmd.acquire_field: taken_at={taken_at}")
            if self.with_agcc_timestamp:
                kwargs["taken_at"] = taken_at  # unix timestamp, not timezone-aware datetime
            if self.with_mlp1_status:
                # possibly override timestamp from agcc
                taken_at = self.actor.mlp1.setUnixDay(telescope_state["az_el_detect_time"], taken_at)
                kwargs["taken_at"] = taken_at
            if center is not None:
                kwargs["center"] = center
            if offset is not None:
                kwargs["offset"] = offset
            if dinr is not None:
                kwargs["dinr"] = dinr
            # retrieve field center coordinates from opdb
            # retrieve exposure information from opdb
            # retrieve guide star coordinates from opdb
            # retrieve metrics of detected objects from opdb
            # compute offsets, scale, transparency, and seeing
            dalt = daz = np.nan
            if guide:
                self.actor.logger.info("AgCmd.acquire_field: guide=True")
                cmd.inform("detectionState=1")
                # convert equatorial coordinates to horizontal coordinates
                self.actor.logger.info(
                    "AgCmd.acquire_field: Calling field_acquisition.acquire_field for guiding"
                )
                guide_offsets = field_acquisition.acquire_field(
                    design=design, frame_id=frame_id, altazimuth=True, logger=self.actor.logger, **kwargs
                )  # design takes precedence over center
                ra = guide_offsets.ra
                dec = guide_offsets.dec
                inst_pa = guide_offsets.inst_pa
                dra = guide_offsets.ra_offset
                ddec = guide_offsets.dec_offset
                dinr = guide_offsets.inr_offset
                dscale = guide_offsets.scale_offset
                dalt = guide_offsets.dalt
                daz = guide_offsets.daz
                guide_files = [
                    guide_offsets.guide_objects,
                    guide_offsets.detected_objects,
                    guide_offsets.identified_objects,
                ]
                cmd.inform(f'text="{ra=},{dec=},{inst_pa=},{dra=},{ddec=},{dinr=},{dscale=},{dalt=},{daz=}"')
                filenames = (
                    "/dev/shm/guide_objects.npy",
                    "/dev/shm/detected_objects.npy",
                    "/dev/shm/identified_objects.npy",
                )
                for filename, value in zip(filenames, guide_files):
                    self.actor.logger.info(f"AgCmd.acquire_field: Saving {filename}")
                    np.save(filename, value)
                cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, inst_pa, *filenames))
                cmd.inform("detectionState=0")
                dx, dy, size, peak, flux = (
                    guide_offsets.dx,
                    guide_offsets.dy,
                    guide_offsets.size,
                    guide_offsets.peak,
                    guide_offsets.flux,
                )
                # send corrections to mlp1 and gen2 (or iic)
                mlp_result = self.actor.queueCommand(
                    actor="mlp1",
                    # daz, dalt: arcsec, positive feedback; dx, dy: mas, HSC -> PFS; size: mas; peak, flux: adu
                    cmdStr="guide azel={},{} ready={} time={} delay=0 xy={},{} size={} intensity={} flux={}".format(
                        -daz,
                        -dalt,
                        int(not dry_run),
                        taken_at,
                        dx * 1e3,
                        -dy * 1e3,
                        size * 13 / 98e-3,
                        peak,
                        flux,
                    ),
                    timeLim=5,
                )
                mlp_result.get()
            else:
                self.actor.logger.info("AgCmd.acquire_field: guide=False")
                cmd.inform("detectionState=1")
                self.actor.logger.info(
                    "AgCmd.acquire_field: Calling field_acquisition.acquire_field not guiding"
                )
                guide_offsets = field_acquisition.acquire_field(
                    design=design, frame_id=frame_id, logger=self.actor.logger, **kwargs
                )  # design takes precedence over center
                ra = guide_offsets.ra
                dec = guide_offsets.dec
                inst_pa = guide_offsets.inst_pa
                dra = guide_offsets.ra_offset
                ddec = guide_offsets.dec_offset
                dinr = guide_offsets.inr_offset
                dscale = guide_offsets.scale_offset
                dalt = guide_offsets.dalt
                daz = guide_offsets.daz
                guide_files = [
                    guide_offsets.guide_objects,
                    guide_offsets.detected_objects,
                    guide_offsets.identified_objects,
                ]
                cmd.inform(f'text="dra={dra},ddec={ddec},dinr={dinr},dscale={dscale}"')
                filenames = (
                    "/dev/shm/guide_objects.npy",
                    "/dev/shm/detected_objects.npy",
                    "/dev/shm/identified_objects.npy",
                )
                for filename, value in zip(filenames, guide_files):
                    self.actor.logger.info(f"AgCmd.acquire_field: Saving {filename}")
                    np.save(filename, value)
                cmd.inform('data={},{},{},"{}","{}","{}"'.format(ra, dec, inst_pa, *filenames))
                cmd.inform("detectionState=0")
                # send corrections to gen2 (or iic)
            # always compute focus offset and tilt
            self.actor.logger.info("AgCmd.acquire_field: Calling focus._focus")
            dz, dzs = _focus._focus(detected_objects=guide_offsets.detected_objects, logger=self.actor.logger)
            # send corrections to gen2 (or iic)
            cmd.inform(
                "guideErrors={},{},{},{},{},{},{},{}".format(frame_id, dra, ddec, dinr, daz, dalt, dz, dscale)
            )
            cmd.inform("focusErrors={},{},{},{},{},{},{}".format(frame_id, *dzs))
            # store results in opdb
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
                    delta_zs=dzs,
                )
            if self.with_opdb_agc_match:
                data_utils.write_agc_match(
                    design_id=(
                        design_id
                        if design_id is not None
                        else pfs_design.pfsDesign.to_design_id(design_path) if design_path is not None else 0
                    ),
                    frame_id=frame_id,
                    guide_objects=guide_offsets.guide_objects,
                    detected_objects=guide_offsets.detected_objects,
                    identified_objects=guide_offsets.identified_objects,
                )
        except Exception as e:
            self.actor.logger.exception("AgCmd.acquire_field:")
            cmd.fail(f'text="AgCmd.acquire_field: {e}"')
            return
        cmd.finish()

    def focus(self, cmd):

        controller = self.actor.controllers["ag"]
        mode = controller.get_mode()
        self.actor.logger.info(f"AgCmd.focus: mode={mode}")
        if mode != controller.Mode.OFF:
            cmd.fail(f'text="AgCmd.focus: mode={mode}"')
            return

        visit_id = None
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
        exposure_time = 2000  # ms
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
        kwargs = {}
        if "max_ellipticity" in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords["max_ellipticity"].values[0])
            kwargs["max_ellipticity"] = max_ellipticity
        if "max_size" in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords["max_size"].values[0])
            kwargs["max_size"] = max_size
        if "min_size" in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords["min_size"].values[0])
            kwargs["min_size"] = min_size
        exposure_delay = ag.EXPOSURE_DELAY
        if "exposure_delay" in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords["exposure_delay"].values[0])
        tec_off = ag.TEC_OFF
        if "tec_off" in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords["tec_off"].values[0])

        try:
            cmd.inform(f"exposureTime={exposure_time}")
            # start an exposure
            cmdStr = f"expose object exptime={exposure_time / 1000} centroid=1"
            if visit_id is not None:
                cmdStr += f" visit={visit_id}"
            if exposure_delay > 0:
                cmdStr += f" threadDelay={exposure_delay}"
            if tec_off:
                cmdStr += " tecOFF"
            agcc_result = self.actor.queueCommand(
                actor="agcc", cmdStr=cmdStr, timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 15)
            )
            # wait for an exposure to complete
            agcc_result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info(f"AgCmd.focus: frameId={frame_id}")
            # retrieve detected objects from agcc (or opdb)
            # compute focus offset and tilt
            dz, dzs = _focus.focus(frame_id=frame_id, logger=self.actor.logger, **kwargs)
            if np.isnan(dz):
                cmd.fail(f'text="AgCmd.focus: dz={dz}"')
                return
            cmd.inform(f'text="dz={dz}"')
            # send corrections to gen2 (or iic)
            cmd.inform(
                "guideErrors={},{},{},{},{},{},{},{}".format(
                    frame_id, np.nan, np.nan, np.nan, np.nan, np.nan, dz, np.nan
                )
            )
            cmd.inform("focusErrors={},{},{},{},{},{},{}".format(frame_id, *dzs))
            # store results in opdb
            if self.with_opdb_agc_guide_offset:
                self.actor.logger.info(f"AgCmd.focus: Writing opdb_agc_guide_offset: {dz=} {dzs=}")
                data_utils.write_agc_guide_offset(frame_id=frame_id, delta_z=dz, delta_zs=dzs)
        except Exception as e:
            self.actor.logger.exception("AgCmd.focus:")
            cmd.fail(f'text="AgCmd.focus: {e}"')
            return
        cmd.finish()

    def start_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.start_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        design_id = None
        if "design_id" in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords["design_id"].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if "design_path" in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords["design_path"].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
        from_sky = None
        if "from_sky" in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords["from_sky"].values[0])
        exposure_time = 2000  # ms
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = 0  # ms
        if "cadence" in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords["cadence"].values[0])
            if cadence < 0:
                cadence = 0
        center = None
        if "center" in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords["center"].values])
        kwargs = {}
        if "magnitude" in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords["magnitude"].values[0])
            kwargs["magnitude"] = magnitude
        if "dry_run" in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords["dry_run"].values[0])
            kwargs["dry_run"] = dry_run
        if "fit_dinr" in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords["fit_dinr"].values[0])
            kwargs["fit_dinr"] = fit_dinr
        if "fit_dscale" in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords["fit_dscale"].values[0])
            kwargs["fit_dscale"] = fit_dscale
        if "max_ellipticity" in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords["max_ellipticity"].values[0])
            kwargs["max_ellipticity"] = max_ellipticity
        if "max_size" in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords["max_size"].values[0])
            kwargs["max_size"] = max_size
        if "min_size" in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords["min_size"].values[0])
            kwargs["min_size"] = min_size
        if "max_residual" in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords["max_residual"].values[0])
            kwargs["max_residual"] = max_residual
        if "exposure_delay" in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords["exposure_delay"].values[0])
            kwargs["exposure_delay"] = exposure_delay
        if "tec_off" in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords["tec_off"].values[0])
            kwargs["tec_off"] = tec_off

        try:
            self.actor.logger.info(f"AgCmd.start_autoguide: kwargs={kwargs}")
            controller.start_autoguide(
                cmd=cmd,
                design=design,
                visit_id=visit_id,
                from_sky=from_sky,
                exposure_time=exposure_time,
                cadence=cadence,
                center=center,
                **kwargs,
            )
        except Exception as e:
            self.actor.logger.exception("AgCmd.start_autoguide:")
            cmd.fail(f'text="AgCmd.start_autoguide: {e}"')
            return
        cmd.finish()

    def initialize_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.initialize_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        design_id = None
        if "design_id" in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords["design_id"].values[0], 0)
        design_path = self.with_design_path if design_id is not None else None
        if "design_path" in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords["design_path"].values[0])
        design = (design_id, design_path) if any(x is not None for x in (design_id, design_path)) else None
        visit_id = None
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
        from_sky = None
        if "from_sky" in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords["from_sky"].values[0])
        exposure_time = 2000  # ms
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
        cadence = 0  # ms
        if "cadence" in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords["cadence"].values[0])
            if cadence < 0:
                cadence = 0
        center = None
        if "center" in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords["center"].values])
        kwargs = {}
        if "magnitude" in cmd.cmd.keywords:
            magnitude = float(cmd.cmd.keywords["magnitude"].values[0])
            kwargs["magnitude"] = magnitude
        if "dry_run" in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords["dry_run"].values[0])
            kwargs["dry_run"] = dry_run
        if "fit_dinr" in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords["fit_dinr"].values[0])
            kwargs["fit_dinr"] = fit_dinr
        if "fit_dscale" in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords["fit_dscale"].values[0])
            kwargs["fit_dscale"] = fit_dscale
        if "max_ellipticity" in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords["max_ellipticity"].values[0])
            kwargs["max_ellipticity"] = max_ellipticity
        if "max_size" in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords["max_size"].values[0])
            kwargs["max_size"] = max_size
        if "min_size" in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords["min_size"].values[0])
            kwargs["min_size"] = min_size
        if "max_residual" in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords["max_residual"].values[0])
            kwargs["max_residual"] = max_residual
        if "exposure_delay" in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords["exposure_delay"].values[0])
            kwargs["exposure_delay"] = exposure_delay
        if "tec_off" in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords["tec_off"].values[0])
            kwargs["tec_off"] = tec_off

        try:
            self.actor.logger.info(f"AgCmd.initialize_autoguide: kwargs={kwargs}")
            controller.initialize_autoguide(
                cmd=cmd,
                design=design,
                visit_id=visit_id,
                from_sky=from_sky,
                exposure_time=exposure_time,
                cadence=cadence,
                center=center,
                **kwargs,
            )
        except Exception as e:
            self.actor.logger.exception("AgCmd.initialize_autoguide:")
            cmd.fail(f'text="AgCmd.initialize_autoguide: {e}"')
            return
        cmd.finish()

    def restart_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.restart_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        try:
            controller.restart_autoguide(cmd=cmd)
        except Exception as e:
            self.actor.logger.exception("AgCmd.restart_autoguide:")
            cmd.fail(f'text="AgCmd.restart_autoguide: {e}"')
            return
        cmd.finish()

    def stop_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.stop_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        try:
            controller.stop_autoguide()
        except Exception as e:
            self.actor.logger.exception("AgCmd.stop_autoguide:")
            cmd.fail(f'text="AgCmd.stop_autoguide: {e}"')
            return
        cmd.finish()

    def reconfigure_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.reconfigure_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        kwargs = {}
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
            kwargs["visit_id"] = visit_id
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
            kwargs["visit_id"] = visit_id
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
            kwargs["exposure_time"] = exposure_time
        if "cadence" in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords["cadence"].values[0])
            if cadence < 0:
                cadence = 0
            kwargs["cadence"] = cadence
        if "dry_run" in cmd.cmd.keywords:
            dry_run = bool(cmd.cmd.keywords["dry_run"].values[0])
            kwargs["dry_run"] = dry_run
        if "fit_dinr" in cmd.cmd.keywords:
            fit_dinr = bool(cmd.cmd.keywords["fit_dinr"].values[0])
            kwargs["fit_dinr"] = fit_dinr
        if "fit_dscale" in cmd.cmd.keywords:
            fit_dscale = bool(cmd.cmd.keywords["fit_dscale"].values[0])
            kwargs["fit_dscale"] = fit_dscale
        if "max_ellipticity" in cmd.cmd.keywords:
            max_ellipticity = float(cmd.cmd.keywords["max_ellipticity"].values[0])
            kwargs["max_ellipticity"] = max_ellipticity
        if "max_size" in cmd.cmd.keywords:
            max_size = float(cmd.cmd.keywords["max_size"].values[0])
            kwargs["max_size"] = max_size
        if "min_size" in cmd.cmd.keywords:
            min_size = float(cmd.cmd.keywords["min_size"].values[0])
            kwargs["min_size"] = min_size
        if "max_residual" in cmd.cmd.keywords:
            max_residual = float(cmd.cmd.keywords["max_residual"].values[0])
            kwargs["max_residual"] = max_residual
        if "exposure_delay" in cmd.cmd.keywords:
            exposure_delay = int(cmd.cmd.keywords["exposure_delay"].values[0])
            kwargs["exposure_delay"] = exposure_delay
        if "tec_off" in cmd.cmd.keywords:
            tec_off = bool(cmd.cmd.keywords["tec_off"].values[0])
            kwargs["tec_off"] = tec_off

        try:
            controller.reconfigure_autoguide(cmd=cmd, **kwargs)
        except Exception as e:
            self.actor.logger.exception("AgCmd.reconfigure_autoguide:")
            cmd.fail(f'text="AgCmd.reconfigure_autoguide: {e}"')
            return
        cmd.finish()

    _SCALE0 = Subaru_POPT2_PFS.Unknown_Scale_Factor_AG

    def offset(self, cmd):
        self.actor.logger.info(f"AgCmd.offset: {cmd.cmd.keywords}")

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
            return (
                Subaru_POPT2_PFS.inr_axis_on_dp_x,
                Subaru_POPT2_PFS.inr_axis_on_dp_y,
                Subaru_POPT2_PFS.inr_zero_offset,
                Subaru_POPT2_PFS.Unknown_Scale_Factor_AG - AgCmd._SCALE0,
            )

        dx = None
        if "dx" in cmd.cmd.keywords:
            dx = float(cmd.cmd.keywords["dx"].values[0])
        dy = None
        if "dy" in cmd.cmd.keywords:
            dy = float(cmd.cmd.keywords["dy"].values[0])
        dinr = None
        if "dinr" in cmd.cmd.keywords:
            dinr = float(cmd.cmd.keywords["dinr"].values[0])
        dscale = None
        if "dscale" in cmd.cmd.keywords:
            dscale = float(cmd.cmd.keywords["dscale"].values[0])
        if "reset" in cmd.cmd.keywords:
            dx, dy, dinr, dscale = 0.0, 0.0, -90.0, 0.0
        dx, dy, dinr, dscale = zero_offset(
            dx=dx, dy=dy, dinr=dinr, dscale=dscale, relative="relative" in cmd.cmd.keywords
        )
        self.actor.logger.info(f"AgCmd.offset: dx={dx},dy={dy},dinr={dinr},dscale={dscale}")
        cmd.inform(f'text="dx={dx},dy={dy},dinr={dinr},dscale={dscale}"')
        cmd.inform(f"guideOffsets={dx},{dy}")
        cmd.finish()
