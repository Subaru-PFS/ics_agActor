from datetime import datetime, timezone
from enum import IntFlag
from numbers import Number

import numpy
from astropy.table import Table
from numpy.lib import recfunctions as rfn
from pfs.utils.coordinates import coordinates

import _gen2_gaia as gaia
import to_altaz
from kawanomoto import FieldAcquisitionAndFocusing
from opdb import opDB as opdb
from pfs_design import pfsDesign as pfs_design

# mapping of keys and value types between field_acquisition.py and FieldAcquisitionAndFocusing.py
_KEYMAP = {
    "fit_dinr": ("inrflag", int),
    "fit_dscale": ("scaleflag", int),
    "max_ellipticity": ("maxellip", float),
    "max_size": ("maxsize", float),
    "min_size": ("minsize", float),
    "max_residual": ("maxresid", float),
}


class AutoGuiderStarMask(IntFlag):
    """
    Represents a bitmask for guide star properties.
    Attributes:
        GAIA: Gaia DR3 catalog.
        HSC: HSC PDR3 catalog.
        PMRA: Proper motion RA is measured.
        PMRA_SIG: Proper motion RA measurement is significant (SNR>5).
        PMDEC: Proper motion Dec is measured.
        PMDEC_SIG: Proper motion Dec measurement is significant (SNR>5).
        PARA: Parallax is measured.
        PARA_SIG: Parallax measurement is significant (SNR>5).
        ASTROMETRIC: Astrometric excess noise is small (astrometric_excess_noise<1.0).
        ASTROMETRIC_SIG: Astrometric excess noise is significant (astrometric_excess_noise_sig>2.0).
        NON_BINARY: Not a binary system (RUWE<1.4).
        PHOTO_SIG: Photometric measurement is significant (SNR>5).
        GALAXY: Is a galaxy candidate.
    """

    GAIA = 0x00001
    HSC = 0x00002
    PMRA = 0x00004
    PMRA_SIG = 0x00008
    PMDEC = 0x00010
    PMDEC_SIG = 0x00020
    PARA = 0x00040
    PARA_SIG = 0x00080
    ASTROMETRIC = 0x00100
    ASTROMETRIC_SIG = 0x00200
    NON_BINARY = 0x00400
    PHOTO_SIG = 0x00800
    GALAXY = 0x01000
    MAX_ELLIPTICITY = 0x02000
    MAX_SIZE = 0x04000
    MIN_SIZE = 0x08000
    MAX_RESID = 0x10000


def _filter_kwargs(kwargs):

    return {k: v for k, v in kwargs.items() if k in _KEYMAP}


def _map_kwargs(kwargs):

    return {_KEYMAP[k][0]: _KEYMAP[k][1](v) for k, v in kwargs.items() if k in _KEYMAP}


def _parse_kwargs(kwargs):

    if (center := kwargs.pop("center", None)) is not None:
        ra, dec, *optional = center
        kwargs.setdefault("ra", ra)
        kwargs.setdefault("dec", dec)
        kwargs.setdefault("inst_pa", optional[0] if len(optional) > 0 else 0)
    if (offset := kwargs.pop("offset", None)) is not None:
        dra, ddec, *optional = offset
        kwargs.setdefault("dra", dra)
        kwargs.setdefault("ddec", ddec)
        kwargs.setdefault("dpa", optional[0] if len(optional) > 0 else kwargs.get("dinr", 0))
        kwargs.setdefault(
            "dinr", optional[1] if len(optional) > 1 else optional[0] if len(optional) > 0 else 0
        )
    if (design := kwargs.pop("design", None)) is not None:
        design_id, design_path = design
        kwargs.setdefault("design_id", design_id)
        kwargs.setdefault("design_path", design_path)
    if (status_id := kwargs.pop("status_id", None)) is not None:
        visit_id, sequence_id = status_id
        kwargs.setdefault("visit_id", visit_id)
        kwargs.setdefault("sequence_id", sequence_id)
    if (tel_status := kwargs.pop("tel_status", None)) is not None:
        _, _, inr, adc, m2_pos3, _, _, _, taken_at = tel_status
        kwargs.setdefault("taken_at", taken_at)
        kwargs.setdefault("inr", inr)
        kwargs.setdefault("adc", adc)
        kwargs.setdefault("m2_pos3", m2_pos3)


def _get_tel_status(*, frame_id, logger=None, **kwargs):
    logger and logger.info("Getting telescope status for frame_id={}".format(frame_id))
    taken_at = kwargs.get("taken_at")
    inr = kwargs.get("inr")
    adc = kwargs.get("adc")
    m2_pos3 = kwargs.get("m2_pos3")
    if any(x is None for x in (taken_at, inr, adc, m2_pos3)):
        logger and logger.info("Getting agc_exposure from opdb for frame_id={}".format(frame_id))
        visit_id, _, _taken_at, _, _, _inr, _adc, _, _, _, _m2_pos3 = opdb.query_agc_exposure(frame_id)
        if (sequence_id := kwargs.get("sequence_id")) is not None:
            logger and logger.info(
                "Getting telescope status from opdb for visit_id={},sequence_id={}".format(
                    visit_id, sequence_id
                )
            )
            # use visit_id from agc_exposure table
            _, _, _inr, _adc, _m2_pos3, _, _, _, _, _taken_at = opdb.query_tel_status(visit_id, sequence_id)
        if taken_at is None:
            taken_at = _taken_at
        if inr is None:
            inr = _inr
        if adc is None:
            adc = _adc
        if m2_pos3 is None:
            m2_pos3 = _m2_pos3
    logger and logger.info("taken_at={},inr={},adc={},m2_pos3={}".format(taken_at, inr, adc, m2_pos3))
    return taken_at, inr, adc, m2_pos3


def acquire_field(*, frame_id, obswl=0.62, altazimuth=False, logger=None, **kwargs):
    logger and logger.info(
        f"Calling acquire_field with frame_id={frame_id}, obswl={obswl}, altazimuth={altazimuth}"
    )
    _parse_kwargs(kwargs)
    taken_at, inr, adc, m2_pos3 = _get_tel_status(frame_id=frame_id, logger=logger, **kwargs)
    logger and logger.info("Getting agc_data from opdb for frame_id={}".format(frame_id))
    detected_objects = opdb.query_agc_data(frame_id)
    logger and logger.info(f"Got {len(detected_objects)=} detected objects)")

    # Check we have detected objects and all flags are <= 1 (right-side flag).
    if len(detected_objects) == 0 and all([d[-1] <= 1 for d in detected_objects]):
        raise RuntimeError("No valid spots detected, can't compute offsets")

    design_id = kwargs.get("design_id")
    design_path = kwargs.get("design_path")
    logger and logger.info("design_id={},design_path={}".format(design_id, design_path))
    ra = kwargs.get("ra")
    dec = kwargs.get("dec")
    inst_pa = kwargs.get("inst_pa")

    if all(x is None for x in (design_id, design_path)):
        logger and logger.info(f"Getting guide objects from gaia db.")
        guide_objects, *_ = gaia.get_objects(
            ra=ra, dec=dec, obstime=taken_at, inst_pa=inst_pa, adc=adc, m2pos3=m2_pos3, obswl=obswl
        )
    else:
        if design_path is not None:
            logger and logger.info(
                f"Getting guide objects from pfs design file via {design_path=} and {design_id=}"
            )
            guide_objects, _ra, _dec, _inst_pa = pfs_design(
                design_id, design_path, logger=logger
            ).guide_objects(obstime=taken_at)
        else:
            logger and logger.info(f"Getting guide objects from opdb via {design_id=}")
            _, _ra, _dec, _inst_pa, *_ = opdb.query_pfs_design(design_id)
            guide_objects = opdb.query_pfs_design_agc(design_id)
        if ra is None:
            ra = _ra
        if dec is None:
            dec = _dec
        if inst_pa is None:
            inst_pa = _inst_pa

    logger and logger.info(f"Got {len(guide_objects)} guide objects before filtering.")
    guide_objects = filter_guide_objects(guide_objects, logger)

    logger and logger.info("ra={},dec={},inst_pa={}".format(ra, dec, inst_pa))

    if "dra" in kwargs:
        ra += kwargs.get("dra") / 3600
    if "ddec" in kwargs:
        dec += kwargs.get("ddec") / 3600
    if "dpa" in kwargs:
        inst_pa += kwargs.get("dpa") / 3600
    if "dinr" in kwargs:
        inr += kwargs.get("dinr") / 3600
    logger and logger.info("ra={},dec={},inst_pa={},inr={}".format(ra, dec, inst_pa, inr))
    _kwargs = _filter_kwargs(kwargs)
    logger and logger.info("Calling _acquire_field with _kwargs={}".format(_kwargs))
    return (
        ra,
        dec,
        inst_pa,
        *_acquire_field(
            guide_objects,
            detected_objects,
            ra,
            dec,
            taken_at,
            adc,
            inst_pa,
            m2_pos3=m2_pos3,
            obswl=obswl,
            altazimuth=altazimuth,
            logger=logger,
            **_kwargs,
        ),
    )


def _acquire_field(
    guide_objects,
    detected_objects,
    ra,
    dec,
    taken_at,
    adc,
    inst_pa=0.0,
    m2_pos3=6.0,
    obswl=0.62,
    altazimuth=False,
    logger=None,
    **kwargs,
):

    def semi_axes(xy, x2, y2):

        p = (x2 + y2) / 2
        q = numpy.sqrt(numpy.square((x2 - y2) / 2) + numpy.square(xy))
        a = numpy.sqrt(p + q)
        b = numpy.sqrt(p - q)
        return a, b

    _guide_objects = numpy.array([(x[1], x[2], x[3]) for x in guide_objects])

    _detected_objects = numpy.array(
        [
            (
                x[0],
                x[1],
                *coordinates.det2dp(int(x[0]), x[3], x[4]),
                x[10],
                *semi_axes(x[5], x[6], x[7]),
                x[-1],
            )
            for x in detected_objects
        ]
    )
    _kwargs = _map_kwargs(kwargs)
    logger and logger.info("Calling pfs.FAinstpa with _kwargs={}".format(_kwargs))
    pfs = FieldAcquisitionAndFocusing.PFS()
    dra, ddec, dinr, dscale, *diags = pfs.FAinstpa(
        _guide_objects,
        _detected_objects,
        ra,
        dec,
        (
            taken_at.astimezone(tz=timezone.utc)
            if isinstance(taken_at, datetime)
            else (
                datetime.fromtimestamp(taken_at, tz=timezone.utc)
                if isinstance(taken_at, Number)
                else taken_at
            )
        ),
        adc,
        inst_pa,
        m2_pos3,
        obswl,
        **_kwargs,
    )
    dra *= 3600
    ddec *= 3600
    dinr *= 3600
    logger and logger.info("From FAinstapa dra={},ddec={},dinr={},dscale={}".format(dra, ddec, dinr, dscale))

    values = ()
    if altazimuth:
        alt, az, dalt, daz = to_altaz.to_altaz(ra, dec, taken_at, dra=dra, ddec=ddec)
        logger and logger.info("alt={},az={},dalt={},daz={}".format(alt, az, dalt, daz))
        values = dalt, daz
    guide_objects = numpy.array(
        [
            (
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                x[6],
                *coordinates.det2dp(x[4], float(x[5]), float(x[6])),
                x[7],  # guide_star_flag
                x[8],  # filter_flag
            )
            for x in guide_objects
        ],
        dtype=[
            ("source_id", numpy.int64),  # u8 (80) not supported by FITSIO
            ("ra", numpy.float64),
            ("dec", numpy.float64),
            ("mag", numpy.float32),
            ("camera_id", numpy.int16),
            ("guide_object_xdet", numpy.float32),
            ("guide_object_ydet", numpy.float32),
            ("guide_object_x", numpy.float32),
            ("guide_object_y", numpy.float32),
            ("guide_star_flag", numpy.int32),  # guide star flag
            ("filter_flag", numpy.int32),  # filter flag
        ],
    )
    detected_objects = numpy.array(
        detected_objects,
        dtype=[
            ("camera_id", numpy.int16),
            ("spot_id", numpy.int16),
            ("moment_00", numpy.float32),
            ("centroid_x", numpy.float32),
            ("centroid_y", numpy.float32),
            ("central_moment_11", numpy.float32),
            ("central_moment_20", numpy.float32),
            ("central_moment_02", numpy.float32),
            ("peak_x", numpy.uint16),
            ("peak_y", numpy.uint16),
            ("peak", numpy.uint16),
            ("background", numpy.float32),
            ("flags", numpy.uint8),
        ],
    )
    mr, md, v = diags
    (index_v,) = numpy.where(v)
    identified_objects = numpy.array(
        [
            (
                k,  # index of detected object
                int(x[0]),  # index of identified guide object
                float(x[1]),
                float(x[2]),  # detector plane coordinates of detected object
                float(x[3]),
                float(x[4]),  # detector plane coordinates of identified guide object
                *coordinates.dp2det(
                    detected_objects[k][0], float(x[3]), float(x[4])
                ),  # detector coordinates of identified guide object
            )
            for k, x in (
                (int(index_v[i]), x)
                for i, x in enumerate(zip(mr[:, 9], mr[:, 0], mr[:, 1], mr[:, 2], mr[:, 3], mr[:, 8]))
                if int(x[5])
            )
        ],
        dtype=[
            ("detected_object_id", numpy.int16),
            ("guide_object_id", numpy.int16),
            ("detected_object_x", numpy.float32),
            ("detected_object_y", numpy.float32),
            ("guide_object_x", numpy.float32),
            ("guide_object_y", numpy.float32),
            ("guide_object_xdet", numpy.float32),
            ("guide_object_ydet", numpy.float32),
        ],
    )

    # convert to arcsec
    logger and logger.info("Converting dra, ddec to arcsec: dra={},ddec={}".format(dra, ddec))
    dx = -dra * numpy.cos(numpy.deg2rad(dec))  # arcsec
    dy = ddec  # arcsec (HSC definition)
    logger and logger.info("dx={},dy={}".format(dx, dy))
    # find "representative" spot size, peak intensity, and flux by "median" of pointing errors
    size = 0  # pix
    peak = 0  # pix
    flux = 0  # pix
    esq = (identified_objects["detected_object_x"] - identified_objects["guide_object_x"]) ** 2 + (
        identified_objects["detected_object_y"] - identified_objects["guide_object_y"]
    ) ** 2  # squares of pointing errors in detector plane coordinates
    n = len(esq) - numpy.isnan(esq).sum()
    if n > 0:
        i = numpy.argpartition(esq, n // 2)[n // 2]  # index of "median" of identified objects
        k = identified_objects["detected_object_id"][i]  # index of "median" of detected objects
        a, b = semi_axes(
            detected_objects["central_moment_11"][k],
            detected_objects["central_moment_20"][k],
            detected_objects["central_moment_02"][k],
        )
        size = (a * b) ** 0.5
        peak = detected_objects["peak"][k]
        flux = detected_objects["moment_00"][k]
    values = *values, guide_objects, detected_objects, identified_objects, dx, dy, size, peak, flux
    return (dra, ddec, dinr, dscale, *values)


def filter_guide_objects(guide_objects, logger, initial=False):
    """Apply filtering to the guide objects based on their flags."""
    # Apply filtering if we have a flag column.
    have_flags = "flag" in guide_objects.dtype.names
    guide_objects_df = None
    if have_flags is True:
        logger.info("Applying filters to guide objects")
        try:
            # Use Table to convert, which handles big-endian and little-endian issues.
            guide_objects_df = Table(guide_objects).to_pandas()
            logger.info(f"Got {len(guide_objects_df)} guide objects.")

            column_names = ["objId", "ra", "dec", "mag", "agId", "agX", "agY", "flag"]
            guide_objects_df.columns = column_names

            # Add a column to indicate which flat was used for filtering.
            guide_objects_df["filtered_by"] = 0

            # Filter the guide objects to only include the ones that are not flagged as galaxies.
            logger.info("Filtering guide objects to remove galaxies.")
            galaxy_idx = (guide_objects_df.flag.values & AutoGuiderStarMask.GALAXY) != 0
            guide_objects_df.loc[galaxy_idx, "filtered_by"] = AutoGuiderStarMask.GALAXY.value
            logger.info(
                f"Filtering by {AutoGuiderStarMask.GALAXY.name}, removes {galaxy_idx.sum()} guide objects."
            )

            # The initial coarse guide uses all the stars and the fine guide uses only the GAIA stars.
            if initial is False:
                filters_for_inclusion = [
                    AutoGuiderStarMask.GAIA,
                    AutoGuiderStarMask.NON_BINARY,
                    AutoGuiderStarMask.ASTROMETRIC,
                    AutoGuiderStarMask.PMRA_SIG,
                    AutoGuiderStarMask.PMDEC_SIG,
                    AutoGuiderStarMask.PARA_SIG,
                    AutoGuiderStarMask.PHOTO_SIG,
                ]

                # Go through the filters and mark which stars would be flagged as NOT meeting the mask requirement.
                for f in filters_for_inclusion:
                    not_filtered = guide_objects_df.filtered_by != AutoGuiderStarMask.GALAXY
                    include_filter = (guide_objects_df.flag.values & f) == 0
                    to_be_filtered = (include_filter & not_filtered) != 0
                    guide_objects_df.loc[to_be_filtered, "filtered_by"] |= f.value
                    logger.info(f"Filtering by {f.name}, removes {to_be_filtered.sum()} guide objects.")

                logger.info(
                    f'After filtering, {len(guide_objects_df.query("filtered_by == 0"))} guide objects remain.'
                )
        except Exception as e:
            logger.warning(f"Error filtering guide objects: {e}")
            logger.info("No filtering applied, using all guide objects.")
    else:
        logger.info("No filtering applied, using all guide objects.")
    # Add the column that indicates what was filtered.
    if guide_objects_df is not None:
        logger.info("Adding filter flag column to guide objects.")
        filterFlag_column = guide_objects_df.filtered_by.to_numpy("<i4")
        filterFlag_column = numpy.array(filterFlag_column, dtype=[("filterFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, filterFlag_column), asrecarray=True, flatten=True)
    else:
        logger.info("No filtering applied, using all guide objects.")
        # If not present, we need to add zero entries for the guide_star_flag and filter_flag.
        guideStarFlag_column = numpy.zeros(len(guide_objects), dtype=[("guideStarFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, guideStarFlag_column), asrecarray=True, flatten=True)

        filterFlag_column = numpy.zeros(len(guide_objects), dtype=[("filterFlag", "<i4")])
        guide_objects = rfn.merge_arrays((guide_objects, filterFlag_column), asrecarray=True, flatten=True)
    return guide_objects
