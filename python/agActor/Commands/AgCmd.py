#!/usr/bin/env python

import numpy as np
import opscore.protocols.keys as keys
import opscore.protocols.types as types
from pfs.utils.coordinates import Subaru_POPT2_PFS
from pfs.utils.database.opdb import OpDB
from pfs.utils.database.gaia import GaiaDB

from agActor.config import AgConfig
from agActor.Controllers.ag import ag
from agActor.exposure import run_exposure_pipeline
from agActor.utils import data as data_utils
from agActor.utils.focus import focus


class AgCmd:
    def __init__(self, actor):
        self.actor = actor
        self.vocab = [
            ("ping", "", self.ping),
            ("status", "", self.status),
            ("show", "", self.show),
            (
                "acquire_field",
                "[<design_id>] "
                "[<design_path>] "
                "[<visit_id>|<visit>] "
                "[<visit0>] "
                "[<exposure_time>] "
                "[<guide>] "
                "[<offset>] "
                "[<dinr>] "
                "[<magnitude>] "
                "[<dry_run>] "
                "[<fit_dinr>] "
                "[<fit_dscale>] "
                "[<max_ellipticity>] "
                "[<max_size>] "
                "[<min_size>] "
                "[<max_residual>] "
                "[<exposure_delay>] "
                "[<tec_off>] "
                "[<filter_bad_shape>]",
                self.acquire_field,
            ),
            (
                "focus",
                "[<visit_id>|<visit>] "
                "[<exposure_time>] "
                "[<max_ellipticity>] "
                "[<max_size>] "
                "[<min_size>] "
                "[<exposure_delay>] "
                "[<tec_off>]",
                self.focus,
            ),
            (
                "autoguide",
                "@start "
                "[<design_id>] "
                "[<design_path>] "
                "[<visit_id>|<visit>] "
                "[<visit0>] "
                "[<from_sky>] "
                "[<exposure_time>] "
                "[<cadence>] "
                "[<center>] "
                "[<magnitude>] "
                "[<dry_run>] "
                "[<fit_dinr>] "
                "[<fit_dscale>] "
                "[<max_ellipticity>] "
                "[<max_size>] "
                "[<min_size>] "
                "[<max_residual>] "
                "[<max_correction>] "
                "[<exposure_delay>] "
                "[<tec_off>]",
                self.start_autoguide,
            ),
            (
                "autoguide",
                "@initialize "
                "[<design_id>] "
                "[<design_path>] "
                "[<visit_id>|<visit>] "
                "[<from_sky>] "
                "[<exposure_time>] "
                "[<cadence>] "
                "[<center>] "
                "[<magnitude>] "
                "[<dry_run>] "
                "[<fit_dinr>] "
                "[<fit_dscale>] "
                "[<max_ellipticity>] "
                "[<max_size>] "
                "[<min_size>] "
                "[<max_residual>] "
                "[<max_correction>] "
                "[<exposure_delay>] "
                "[<tec_off>]",
                self.initialize_autoguide,
            ),
            ("autoguide", "@restart", self.restart_autoguide),
            ("autoguide", "@stop", self.stop_autoguide),
            (
                "autoguide",
                "@reconfigure "
                "[<visit_id>|<visit>] "
                "[<exposure_time>] "
                "[<cadence>] "
                "[<dry_run>] "
                "[<fit_dinr>] "
                "[<fit_dscale>] "
                "[<max_ellipticity>] "
                "[<max_size>] "
                "[<min_size>] "
                "[<max_residual>] "
                "[<max_correction>] "
                "[<exposure_delay>] "
                "[<tec_off>] "
                "[<filter_bad_shape>]",
                self.reconfigure_autoguide,
            ),
            (
                "offset",
                "[@(absolute|relative)] [<dx>] [<dy>] [<dinr>] [<dscale>]",
                self.offset,
            ),
            ("offset", "@reset", self.offset),
        ]
        self.keys = keys.KeysDictionary(
            "ag_ag",
            (1, 18),
            keys.Key("exposure_time", types.Int(), help=""),
            keys.Key("cadence", types.Int(), help=""),
            keys.Key("guide", types.Bool("no", "yes"), help=""),
            keys.Key("design_id", types.String(), help=""),
            keys.Key("design_path", types.String(), help=""),
            keys.Key("visit_id", types.Int(), help=""),
            keys.Key("visit", types.Int(), help=""),
            keys.Key("visit0", types.Int(), help="The visit0 associated with a pfsConfig"),
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
            keys.Key("max_correction", types.Float(), help=""),
            keys.Key("exposure_delay", types.Int(), help=""),
            keys.Key("tec_off", types.Bool("no", "yes"), help=""),
            keys.Key("filter_bad_shape", types.Bool("no", "yes"), help=""),
        )

        # Set up the database connections.
        db_params = actor.actorConfig.get("db", {})
        self.actor.logger.info(f"AgCmd: Setting default db_params={db_params}")
        OpDB.set_default_connection(**db_params.get("opdb", {}))
        GaiaDB.set_default_connection(**db_params.get("gaia", {}))

        # Parse shared configuration once and store on the actor for AgThread to use.
        self.cfg = AgConfig.from_actor_config(actor.actorConfig)
        actor.ag_config = self.cfg
        self.actor.logger.info(f"AgCmd: ag_config={self.cfg}")

    def _parse_design(self, cmd):
        """Parse design_id/design_path keywords into a design tuple or None."""
        design_id = None
        if "design_id" in cmd.cmd.keywords:
            design_id = int(cmd.cmd.keywords["design_id"].values[0], 0)
        design_path = self.cfg.with_design_path if design_id is not None else None
        if "design_path" in cmd.cmd.keywords:
            design_path = str(cmd.cmd.keywords["design_path"].values[0])
        design = (
            (design_id, design_path)
            if any(x is not None for x in (design_id, design_path))
            else None
        )
        return design_id, design_path, design

    def _parse_visit(self, cmd):
        """Parse visit_id/visit keywords."""
        visit_id = None
        if "visit_id" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit_id"].values[0])
        elif "visit" in cmd.cmd.keywords:
            visit_id = int(cmd.cmd.keywords["visit"].values[0])
        return visit_id

    def _parse_exposure_time(self, cmd, default=2000):
        """Parse exposure_time keyword with a minimum of 100 ms."""
        exposure_time = default
        if "exposure_time" in cmd.cmd.keywords:
            exposure_time = int(cmd.cmd.keywords["exposure_time"].values[0])
            if exposure_time < 100:
                exposure_time = 100
        return exposure_time

    def _parse_cadence(self, cmd, default=0):
        """Parse cadence keyword with a minimum of 0 ms."""
        cadence = default
        if "cadence" in cmd.cmd.keywords:
            cadence = int(cmd.cmd.keywords["cadence"].values[0])
            if cadence < 0:
                cadence = 0
        return cadence

    # Mapping of keyword name -> (type_converter, default_value_or_None).
    # Keywords with a default of None are only added to the options dict when present.
    _OPTION_KEYWORDS = {
        "magnitude": (float, None),
        "dry_run": (bool, None),
        "fit_dinr": (bool, None),
        "fit_dscale": (bool, None),
        "max_ellipticity": (float, None),
        "max_size": (float, None),
        "min_size": (float, None),
        "max_residual": (float, None),
        "max_correction": (float, None),
        "exposure_delay": (int, None),
        "tec_off": (bool, None),
        "filter_bad_shape": (bool, None),
    }

    def _parse_options(self, cmd):
        """Parse optional keywords that map into the options/kwargs dict.

        Only keywords that are present in the command are included in the
        returned dict, so callers can distinguish "not provided" from an
        explicit value.
        """
        options = {}
        for key, (converter, _) in self._OPTION_KEYWORDS.items():
            if key in cmd.cmd.keywords:
                options[key] = converter(cmd.cmd.keywords[key].values[0])
        return options

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

        design_id, design_path, design = self._parse_design(cmd)
        visit_id = self._parse_visit(cmd)

        visit0 = None
        if "visit0" in cmd.cmd.keywords:
            visit0 = int(cmd.cmd.keywords["visit0"].values[0])

        exposure_time = self._parse_exposure_time(cmd)

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

        kwargs = self._parse_options(cmd)

        # acquire_field-specific defaults for options not provided by the user.
        dry_run = kwargs.pop("dry_run", ag.DRY_RUN)
        max_ellipticity = kwargs.get("max_ellipticity", ag.MAX_ELLIPTICITY)
        max_size = kwargs.get("max_size", ag.MAX_SIZE)
        min_size = kwargs.get("min_size", ag.MIN_SIZE)
        exposure_delay = kwargs.pop("exposure_delay", ag.EXPOSURE_DELAY)
        tec_off = kwargs.pop("tec_off", ag.TEC_OFF)
        kwargs.setdefault("filter_bad_shape", ag.FILTER_BAD_SHAPE)

        self.actor.logger.info(f"AgCmd.acquire_field: kwargs={kwargs}")

        try:
            run_exposure_pipeline(
                actor=self.actor,
                cmd=cmd,
                cfg=self.cfg,
                design_id=design_id,
                design_path=design_path,
                design=design,
                visit_id=visit_id,
                visit0=visit0,
                exposure_time=exposure_time,
                exposure_delay=exposure_delay,
                tec_off=tec_off,
                center=center,
                offset=offset,
                dinr=dinr,
                guide_catalog=None,
                send_offsets=guide,
                dry_run=dry_run,
                max_correction=None,  # no range checking for acquire_field
                max_ellipticity=max_ellipticity,
                max_size=max_size,
                min_size=min_size,
                **kwargs,
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

        visit_id = self._parse_visit(cmd)
        exposure_time = self._parse_exposure_time(cmd)

        options = self._parse_options(cmd)
        max_ellipticity = options.get("max_ellipticity", ag.MAX_ELLIPTICITY)
        max_size = options.get("max_size", ag.MAX_SIZE)
        min_size = options.get("min_size", ag.MIN_SIZE)
        exposure_delay = options.get("exposure_delay", ag.EXPOSURE_DELAY)
        tec_off = options.get("tec_off", ag.TEC_OFF)

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
                actor="agcc",
                cmdStr=cmdStr,
                timeLim=((exposure_time + 6 * exposure_delay) // 1000 + 15),
            )
            # wait for an exposure to complete
            agcc_result.get()
            frame_id = self.actor.agcc.frameId
            self.actor.logger.info(f"AgCmd.focus: frameId={frame_id}")

            # compute focus offset and tilt
            dz, dzs = focus(
                frame_id=frame_id,
                max_ellipticity=max_ellipticity,
                max_size=max_size,
                min_size=min_size,
            )
            if np.isnan(dz):
                cmd.fail(f'text="AgCmd.focus: dz={dz}"')
                return
            cmd.inform(f'text="dz={dz}"')
            # send corrections to gen2 (or iic)
            guide_status = "OK"
            cmd.inform(
                "guideErrors={},{},{},{},{},{},{},{},{}".format(
                    frame_id,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    dz,
                    np.nan,
                    guide_status,
                )
            )
            cmd.inform("focusErrors={},{},{},{},{},{},{}".format(frame_id, *dzs))
            # store results in opdb
            if self.cfg.with_opdb_agc_guide_offset:
                self.actor.logger.info(
                    f"AgCmd.focus: Writing opdb_agc_guide_offset: {dz=} {dzs=}"
                )
                data_utils.write_agc_guide_offset(
                    frame_id=frame_id, delta_z=dz, delta_zs=dzs
                )
        except Exception as e:
            self.actor.logger.exception("AgCmd.focus:")
            cmd.fail(f'text="AgCmd.focus: {e}"')
            return
        cmd.finish()

    def start_autoguide(self, cmd):
        self.actor.logger.info(f"AgCmd.start_autoguide: {cmd.cmd.keywords}")
        controller = self.actor.controllers["ag"]

        design_id, design_path, design = self._parse_design(cmd)
        visit_id = self._parse_visit(cmd)

        visit0 = None
        if "visit0" in cmd.cmd.keywords:
            visit0 = int(cmd.cmd.keywords["visit0"].values[0])

        from_sky = None
        if "from_sky" in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords["from_sky"].values[0])
        exposure_time = self._parse_exposure_time(cmd)
        cadence = self._parse_cadence(cmd)
        center = None
        if "center" in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords["center"].values])

        kwargs = self._parse_options(cmd)

        try:
            self.actor.logger.info(f"AgCmd.start_autoguide: kwargs={kwargs}")
            controller.start_autoguide(
                cmd=cmd,
                design=design,
                visit_id=visit_id,
                visit0=visit0,
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

        design_id, design_path, design = self._parse_design(cmd)
        visit_id = self._parse_visit(cmd)

        from_sky = None
        if "from_sky" in cmd.cmd.keywords:
            from_sky = bool(cmd.cmd.keywords["from_sky"].values[0])
        exposure_time = self._parse_exposure_time(cmd)
        cadence = self._parse_cadence(cmd)
        center = None
        if "center" in cmd.cmd.keywords:
            center = tuple([float(x) for x in cmd.cmd.keywords["center"].values])

        kwargs = self._parse_options(cmd)

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
        visit_id = self._parse_visit(cmd)
        if visit_id is not None:
            kwargs["visit_id"] = visit_id
        if "exposure_time" in cmd.cmd.keywords:
            kwargs["exposure_time"] = self._parse_exposure_time(cmd)
        if "cadence" in cmd.cmd.keywords:
            kwargs["cadence"] = self._parse_cadence(cmd)

        kwargs.update(self._parse_options(cmd))

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
            dx=dx,
            dy=dy,
            dinr=dinr,
            dscale=dscale,
            relative="relative" in cmd.cmd.keywords,
        )
        self.actor.logger.info(
            f"AgCmd.offset: dx={dx},dy={dy},dinr={dinr},dscale={dscale}"
        )
        cmd.inform(f'text="dx={dx},dy={dy},dinr={dinr},dscale={dscale}"')
        cmd.inform(f"guideOffsets={dx},{dy}")
        cmd.finish()
