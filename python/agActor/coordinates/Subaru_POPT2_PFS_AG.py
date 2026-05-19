import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from pfs.utils.coordinates import Subaru_POPT2_PFS as pfs
from pfs.utils.datamodel.ag import SourceDetectionFlags

logger = logging.getLogger(__name__)

### perturbation
d_ra  = 1.0/3600.0
d_de  = 1.0/3600.0
d_inr = 0.01
d_scl = 1.0e-05


class PFS():
    def _build_coefficients(self, ra_coeff, de_coeff, fit_inr, fit_scale):
        """Build a least-squares coefficient vector for the active fit terms.

        Slots for RA and Dec are set to the supplied coarse estimates.  Any
        InR and scale slots are filled with NaN so that the caller's unit-
        conversion step produces NaN for those offsets (matching the
        convention used when fit_inr / fit_scale are False).
        """
        coeff_count = 2 + int(fit_inr) + int(fit_scale)
        coeffs = np.full((coeff_count, 1), np.nan)
        coeffs[0, 0] = ra_coeff
        coeffs[1, 0] = de_coeff
        return coeffs

    def _distance_summary(self, distances):
        """Return min/median/max for a 1-D array, ignoring non-finite values."""
        finite = np.asarray(distances)[np.isfinite(distances)]
        if finite.size == 0:
            return np.nan, np.nan, np.nan
        return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))

    def sourceFilter(self, agarray, maxellip, maxsize, minsize):
        """Filter detected AG sources by shape quality criteria.

        Notes
        -----
        This method is essentially not used anymore because we always call it with
        such permissive values for each of the respective filters.  The real filtering
        logic is done further up the chain and uses the `filtered_by` column.

        Parameters
        ----------
        agarray : np.ndarray, shape (N, 8)
            Array of detected AG sources. Columns are:
              0: ccd       — CCD/camera index (1-based integer)
              1: id        — source ID within the frame
              2: xc [mm]   — centroid x position on the detector plane
              3: yc [mm]   — centroid y position on the detector plane
              4: flux      — integrated source flux [counts]
              5: semi-major [px] — semi-major axis of fitted ellipse
              6: semi-minor [px] — semi-minor axis of fitted ellipse
              7: flag      — detection flag (see SourceDetectionFlags)
        maxellip : float
            Maximum allowed ellipticity ``1 - b/a``.  Sources with a
            rounder PSF (smaller ellipticity) are retained.
        maxsize : float
            Maximum allowed geometric-mean PSF radius
            ``sqrt(a * b)`` [pixels].  Sources larger than this are
            rejected (e.g. extended or blended objects).
        minsize : float
            Minimum allowed geometric-mean PSF radius [pixels].
            Sources smaller than this are rejected (e.g. cosmic rays).

        Returns
        -------
        oarray : np.ndarray, shape (M, 8)
            Subset of ``agarray`` containing only the M sources that
            pass all three criteria.  Column layout is identical to the
            input.
        valid_mask : np.ndarray, shape (N,), dtype bool
            Boolean mask over the original ``agarray`` rows; ``True``
            where the source passed every filter criterion.
        """
        ag_ccd  = agarray[:,0]
        ag_id   = agarray[:,1]
        ag_xc   = agarray[:,2]
        ag_yc   = agarray[:,3]
        ag_flx  = agarray[:,4]
        ag_smma = agarray[:,5]
        ag_smmi = agarray[:,6]
        ag_flag = agarray[:,7]

        # ellipticity condition
        cellip = (1.0-ag_smmi/ag_smma)    < maxellip
        # size condition (upper)
        csizeU = np.sqrt(ag_smmi*ag_smma) < maxsize
        # size condition (lower)
        csizeL = np.sqrt(ag_smmi*ag_smma) > minsize

        valid_mask = cellip*csizeU*csizeL

        oarray = np.zeros((valid_mask.sum(),8))
        oarray[:,0] = ag_ccd[valid_mask]
        oarray[:,1] = ag_id[valid_mask]
        oarray[:,2] = ag_xc[valid_mask]
        oarray[:,3] = ag_yc[valid_mask]
        oarray[:,4] = ag_flx[valid_mask]
        oarray[:,5] = ag_smma[valid_mask]
        oarray[:,6] = ag_smmi[valid_mask]
        oarray[:,7] = ag_flag[valid_mask]

        return oarray, valid_mask


    def RADECInRShiftA(self, obj_xdp, obj_ydp, obj_int, obj_flag,
                       catalog_left, catalog_right,
                       fit_inr: bool, fit_scale: bool, maxresid=0.5,
                       obj_camera_id=None, enabled_camera_ids=None):
        """Perform a full astrometric solve: match detected sources to a catalog
        and determine the pointing offsets (RA, Dec, InR, scale).

        The solve proceeds in six algorithmic phases:
          1. Catalog unpacking
          2. Jacobian averaging
          3. Coarse nearest-neighbour match + Cramer's rule
          4. Refined nearest-neighbour match
          5. Least-squares solve + initial residuals
          6. Iterative outlier rejection
          7. Unit conversion

        Notes
        -----
        In practice we always set `fit_inr` and `fit_scale` to `True` at acquisition,
        so the returned `inr_offset` and `scale_offset` are always valid.  The
        option to disable either fit is available for testing and experimentation,
        but it is not currently used in the standard acquisition sequence.

        Parameters
        ----------
        obj_xdp : np.ndarray, shape (N,)
            Detected source x positions on the detector plane [mm].
        obj_ydp : np.ndarray, shape (N,)
            Detected source y positions on the detector plane [mm].
        obj_int : np.ndarray, shape (N,)
            Detected source intensities / fluxes [counts].  Accepted but
            currently unused in the fitting.
        obj_flag : np.ndarray, shape (N,), dtype int
            Per-source detection flags (see ``SourceDetectionFlags``).
            The ``RIGHT`` bit is used to select which detector-half
            catalog (``catalog_left`` or ``catalog_right``) applies to
            each source.
        catalog_left : np.ndarray, shape (M, 9)
            Catalog basis array for the LEFT detector half.  Each row
            corresponds to one catalog star and contains:
              col 0: xdp  — detector-plane x position [mm]
              col 1: ydp  — detector-plane y position [mm]
              col 2: mag  — magnitude (inserted by the caller via
                            ``np.insert`` before this method is called)
              col 3: dx/dRA   — ∂x_dp/∂(RA perturbation)
              col 4: dy/dRA   — ∂y_dp/∂(RA perturbation)
              col 5: dx/dDec  — ∂x_dp/∂(Dec perturbation)
              col 6: dy/dDec  — ∂y_dp/∂(Dec perturbation)
              col 7: dx/dInR  — ∂x_dp/∂(InR perturbation)
              col 8: dy/dInR  — ∂y_dp/∂(InR perturbation)
            Scale Jacobians (dxscl, dyscl) are derived internally as
            ``xdp * d_scl`` and ``ydp * d_scl`` (pure radial stretch model).
        catalog_right : np.ndarray, shape (M, 9)
            Catalog basis array for the RIGHT detector half.  Same
            column layout as ``catalog_left``.
        fit_inr : bool
            ``True`` to include instrument rotation (InR) in the fit,
            ``False`` to fix InR at its current value
        fit_scale : bool
            ``True`` to include a radial scale term in the fit,
            ``False`` to omit it.
        maxresid : float, optional
            Hard upper limit on the outlier-rejection threshold [mm].
            Defaults to 0.5 mm.
        obj_camera_id : np.ndarray, shape (N,), dtype int, optional
            Camera / CCD index (1-based) for each detected source.  Must be
            provided when ``enabled_camera_ids`` is not ``None``.  When
            ``None`` all sources are treated as coming from an enabled camera.
        enabled_camera_ids : list[int] or None, optional
            Camera IDs whose detections are allowed to contribute to the
            coarse and least-squares solves (Phases 3–6).  Detections from
            cameras *not* in this list are still matched and appear in
            ``match_result``, but they never enter the fit.  When ``None``
            (the default) all cameras are used, preserving the original
            behaviour.  If the enabled set is empty, the coarse solve falls
            back to all detections and logs a warning.  If the refined fit has
            no usable rows, the method falls back to the coarse RA/Dec solution
            and logs a warning.

        Returns
        -------
        ra_offset : float
            Right-ascension offset [degrees].
        de_offset : float
            Declination offset [degrees].
        inr_offset : float or nan
            Instrument-rotation offset [degrees].  ``nan`` when
            ``fit_inr`` is ``False``.
        scale_offset : float or nan
            Radial scale offset [dimensionless].  ``nan`` when
            ``fit_scale`` is ``False``.
        match_result : np.ndarray, shape (N, 11)
            Per-match result matrix with columns:
              0:  obj_x   — detected object x [mm]
              1:  obj_y   — detected object y [mm]
              2:  cat_x   — matched catalog x [mm]
              3:  cat_y   — matched catalog y [mm]
              4:  err_x   — positional error x [mm]
              5:  err_y   — positional error y [mm]
              6:  resid_x — post-fit residual x [mm]
              7:  resid_y — post-fit residual y [mm]
              8:  is_inlier — 1.0 if the match survived outlier rejection
              9:  cat_index — index of the matched star in the catalog arrays
              10: camera_enabled — 1.0 if this detection's camera is in
                  enabled_camera_ids, 0.0 otherwise.  Note: a detection
                  from an enabled camera may still not enter the fit if it
                  had no close catalog match (distance > 2 mm).
        """

        # ── Phase 1: Catalog unpacking ────────────────────────────────────────
        # Extract detector-plane positions, magnitudes, and RA/Dec/InR Jacobians
        # from the left (catalog_left) and right (catalog_right) detector-half
        # catalog basis arrays.  Scale Jacobians are not stored in the basis;
        # they are derived here from the catalog position multiplied by d_scl
        # (pure radial stretch model).
        cat_xdp_0 = catalog_left[:,0]
        cat_ydp_0 = catalog_left[:,1]
        cat_mag_0 = catalog_left[:,2]
        dxra_0    = catalog_left[:,3]
        dyra_0    = catalog_left[:,4]
        dxde_0    = catalog_left[:,5]
        dyde_0    = catalog_left[:,6]
        dxinr_0   = catalog_left[:,7]
        dyinr_0   = catalog_left[:,8]
        dxscl_0   = catalog_left[:,0]*d_scl
        dyscl_0   = catalog_left[:,1]*d_scl

        cat_xdp_1 = catalog_right[:,0]
        cat_ydp_1 = catalog_right[:,1]
        cat_mag_1 = catalog_right[:,2]
        dxra_1    = catalog_right[:,3]
        dyra_1    = catalog_right[:,4]
        dxde_1    = catalog_right[:,5]
        dyde_1    = catalog_right[:,6]
        dxinr_1   = catalog_right[:,7]
        dyinr_1   = catalog_right[:,8]
        dxscl_1   = catalog_right[:,0]*d_scl
        dyscl_1   = catalog_right[:,1]*d_scl

        # ── Phase 2: Jacobian averaging ───────────────────────────────────────
        # Average the left- and right-half Jacobians to obtain a single set of
        # derivatives that is used during the coarse matching step.  This gives
        # a half-detector-agnostic linear sensitivity that is good enough for
        # the initial 2×2 Cramer's-rule solve.
        dxra  = (dxra_0  + dxra_1 )/2.0
        dyra  = (dyra_0  + dyra_1 )/2.0
        dxde  = (dxde_0  + dxde_1 )/2.0
        dyde  = (dyde_0  + dyde_1 )/2.0

        # ── Phase 3: Coarse nearest-neighbour match + Cramer's rule ──────────
        # For each detected object find its nearest catalog star, then solve
        # the 2×2 linear system:
        #   [dxra  dxde] [coarse_ra_coeff]   [Δx]
        #   [dyra  dyde] [coarse_dec_coeff] = [Δy]
        # analytically via Cramer's rule to get a median RA/Dec-only initial
        # offset in perturbation units.  Note: InR/scale errors are not
        # corrected here — they typically dominate only at fine-guiding level,
        # not at acquisition (see plan note B5).
        if enabled_camera_ids is not None and obj_camera_id is not None:
            enabled_mask = np.isin(obj_camera_id, enabled_camera_ids)
        else:
            enabled_mask = np.ones(len(obj_xdp), dtype=bool)

        coarse_mask = enabled_mask
        if enabled_camera_ids is not None and not np.any(coarse_mask):
            logger.warning(
                "RADECInRShiftA coarse solve had no enabled detections; falling back to all detections. "
                "n_obj=%d enabled_camera_ids=%s",
                len(obj_xdp),
                enabled_camera_ids,
            )
            coarse_mask = np.ones(len(obj_xdp), dtype=bool)

        coarse_obj_xdp = obj_xdp[coarse_mask]
        coarse_obj_ydp = obj_ydp[coarse_mask]
        coarse_obj_flag = obj_flag[coarse_mask]
        coarse_right_detector_mask = np.where(
            (coarse_obj_flag.astype(int) & SourceDetectionFlags.RIGHT) == SourceDetectionFlags.RIGHT
        )

        coarse_n_obj = (coarse_obj_xdp.shape)[0]

        xdiff_0 = np.transpose([coarse_obj_xdp]) - cat_xdp_0
        ydiff_0 = np.transpose([coarse_obj_ydp]) - cat_ydp_0
        xdiff_1 = np.transpose([coarse_obj_xdp]) - cat_xdp_1
        ydiff_1 = np.transpose([coarse_obj_ydp]) - cat_ydp_1

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[coarse_right_detector_mask] = xdiff_1[coarse_right_detector_mask]
        ydiff[coarse_right_detector_mask] = ydiff_1[coarse_right_detector_mask]

        dist = np.sqrt(xdiff**2 + ydiff**2)

        min_dist_index = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(coarse_n_obj), dtype="int"), min_dist_index
        coarse_distances = dist[min_dist_indices]
        coarse_min, coarse_median, coarse_max = self._distance_summary(coarse_distances)
        # Cramer's-rule solution for the RA/Dec perturbation coefficients.
        # coarse_ra_coeff and coarse_dec_coeff are still in perturbation units
        # (not arcsec).
        coarse_ra_coeff = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        coarse_dec_coeff = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))
        logger.info(
            "RADECInRShiftA coarse stats: n_obj=%d n_enabled=%d coarse_mm[min/med/max]=%.3f/%.3f/%.3f enabled_camera_ids=%s",
            coarse_n_obj,
            int(np.count_nonzero(coarse_mask)),
            coarse_min,
            coarse_median,
            coarse_max,
            enabled_camera_ids,
        )

        # ── Phase 4: Refined nearest-neighbour match ──────────────────────────
        # Apply the coarse RA/Dec offset to shift the catalog positions, then
        # redo the nearest-neighbour search.  Only pairs within 2 mm of each
        # other are accepted for the least-squares solve.
        n_obj = (obj_xdp.shape)[0]
        xdiff_0 = np.transpose([obj_xdp])-(cat_xdp_0+coarse_ra_coeff*dxra+coarse_dec_coeff*dxde)
        ydiff_0 = np.transpose([obj_ydp])-(cat_ydp_0+coarse_ra_coeff*dyra+coarse_dec_coeff*dyde)

        xdiff_1 = np.transpose([obj_xdp])-(cat_xdp_1+coarse_ra_coeff*dxra+coarse_dec_coeff*dxde)
        ydiff_1 = np.transpose([obj_ydp])-(cat_ydp_1+coarse_ra_coeff*dyra+coarse_dec_coeff*dyde)

        # Re-derive the RIGHT-detector mask from the full obj_flag here.
        # coarse_right_detector_mask has indices into the coarse (enabled-only)
        # subarray and must NOT be reused on the full (n_obj) arrays below.
        right_detector_mask = np.where(
            (obj_flag.astype(int) & SourceDetectionFlags.RIGHT) == SourceDetectionFlags.RIGHT
        )

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[right_detector_mask] = xdiff_1[right_detector_mask]
        ydiff[right_detector_mask] = ydiff_1[right_detector_mask]

        dist  = np.sqrt(xdiff**2+ydiff**2)

        # Enforce one-to-one matching using the Hungarian algorithm (Linear Sum Assignment).
        # This prevents multiple detections from claiming the same catalog star,
        # which can skew the fit and cause widespread camera rejection when transients
        # appear near real stars (many-to-one matching bug).
        cost = np.copy(dist)
        # Replace NaN with a very large penalty to ensure they are only matched as a last resort.
        cost[np.isnan(cost)] = 1000.0

        row_ind, col_ind = linear_sum_assignment(cost)

        # We need a valid catalog index for every detection to avoid indexing errors,
        # even for unmatched ones. We use nanargmin as a baseline and overwrite
        # with the optimal one-to-one assignments.
        min_dist_index = np.nanargmin(dist, axis=1)

        # Boolean mask for detections that received an optimal one-to-one assignment.
        # Detections not in this mask are "extra" and are rejected from the fit.
        assignment_mask = np.zeros(n_obj, dtype=bool)
        assignment_mask[row_ind] = True
        min_dist_index[row_ind] = col_ind

        min_dist_indices = np.array(range(n_obj), dtype='int'),min_dist_index

        # Boolean mask: True where the refined match distance is within 2 mm AND
        # it was the unique optimal assignment for that catalog star.
        close_match_mask  = (dist[min_dist_indices] < 2.0) & assignment_mask
        nearest_distances = dist[min_dist_indices]

        match_obj_xdp  = obj_xdp
        match_obj_ydp  = obj_ydp
        match_obj_flag = obj_flag

        match_cat_xdp_0 = (cat_xdp_0[min_dist_index])
        match_cat_ydp_0 = (cat_ydp_0[min_dist_index])
        match_cat_mag_0 = (cat_mag_0[min_dist_index])
        match_dxra_0    = (dxra_0[min_dist_index])
        match_dyra_0    = (dyra_0[min_dist_index])
        match_dxde_0    = (dxde_0[min_dist_index])
        match_dyde_0    = (dyde_0[min_dist_index])
        match_dxinr_0   = (dxinr_0[min_dist_index])
        match_dyinr_0   = (dyinr_0[min_dist_index])
        match_dxscl_0   = (dxscl_0[min_dist_index])
        match_dyscl_0   = (dyscl_0[min_dist_index])

        match_cat_xdp_1 = (cat_xdp_1[min_dist_index])
        match_cat_ydp_1 = (cat_ydp_1[min_dist_index])
        match_cat_mag_1 = (cat_mag_1[min_dist_index])
        match_dxra_1    = (dxra_1[min_dist_index])
        match_dyra_1    = (dyra_1[min_dist_index])
        match_dxde_1    = (dxde_1[min_dist_index])
        match_dyde_1    = (dyde_1[min_dist_index])
        match_dxinr_1   = (dxinr_1[min_dist_index])
        match_dyinr_1   = (dyinr_1[min_dist_index])
        match_dxscl_1   = (dxscl_1[min_dist_index])
        match_dyscl_1   = (dyscl_1[min_dist_index])

        # Select the appropriate detector-half catalog values for each source
        # based on the RIGHT flag in obj_flag.
        match_cat_xdp = np.copy(match_cat_xdp_0)
        match_cat_ydp = np.copy(match_cat_ydp_0)
        match_cat_mag = np.copy(match_cat_mag_0)
        match_dxra    = np.copy(match_dxra_0)
        match_dyra    = np.copy(match_dyra_0)
        match_dxde    = np.copy(match_dxde_0)
        match_dyde    = np.copy(match_dyde_0)
        match_dxinr   = np.copy(match_dxinr_0)
        match_dyinr   = np.copy(match_dyinr_0)
        match_dxscl   = np.copy(match_dxscl_0)
        match_dyscl   = np.copy(match_dyscl_0)

        right_detector_mask = np.where((match_obj_flag.astype(int) & SourceDetectionFlags.RIGHT) == SourceDetectionFlags.RIGHT)

        match_cat_xdp[right_detector_mask] = match_cat_xdp_1[right_detector_mask]
        match_cat_ydp[right_detector_mask] = match_cat_ydp_1[right_detector_mask]
        match_cat_mag[right_detector_mask] = match_cat_mag_1[right_detector_mask]
        match_dxra[right_detector_mask]    = match_dxra_1[right_detector_mask]
        match_dyra[right_detector_mask]    = match_dyra_1[right_detector_mask]
        match_dxde[right_detector_mask]    = match_dxde_1[right_detector_mask]
        match_dyde[right_detector_mask]    = match_dyde_1[right_detector_mask]
        match_dxinr[right_detector_mask]   = match_dxinr_1[right_detector_mask]
        match_dyinr[right_detector_mask]   = match_dyinr_1[right_detector_mask]
        match_dxscl[right_detector_mask]   = match_dxscl_1[right_detector_mask]
        match_dyscl[right_detector_mask]   = match_dyscl_1[right_detector_mask]

        # ── Phase 5: Least-squares solve + initial residuals ─────────────────
        # Build the design matrix (basis) by stacking the x and y Jacobian
        # columns for each enabled degree of freedom.  Only close-matched
        # pairs (close_match_mask) enter the initial solve.
        fit_mask = close_match_mask & enabled_mask
        n_enabled = int(np.count_nonzero(enabled_mask))
        n_close = int(np.count_nonzero(close_match_mask))
        n_fit = int(np.count_nonzero(fit_mask))
        close_min, close_median, close_max = self._distance_summary(nearest_distances)
        logger.info(
            "RADECInRShiftA fit stats: n_obj=%d n_enabled=%d n_close=%d n_fit=%d "
            "nearest_mm[min/med/max]=%.3f/%.3f/%.3f enabled_camera_ids=%s",
            n_obj,
            n_enabled,
            n_close,
            n_fit,
            close_min,
            close_median,
            close_max,
            enabled_camera_ids,
        )

        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])
        dscl = np.concatenate([match_dxscl,match_dyscl])

        if fit_inr == 1 and fit_scale == 1:
            basis= np.stack([dra,dde,dinr,dscl]).transpose()
        elif fit_inr == 1 and fit_scale == 0:
            basis= np.stack([dra,dde,dinr]).transpose()
        elif fit_inr == 0 and fit_scale == 1:
            basis= np.stack([dra,dde,dscl]).transpose()
        else:
            basis= np.stack([dra,dde]).transpose()

        errx = match_obj_xdp - match_cat_xdp
        erry = match_obj_ydp - match_cat_ydp
        err  = np.array([np.concatenate([errx,erry])]).transpose()

        if n_fit == 0:
            logger.warning(
                "RADECInRShiftA refined fit had no usable rows; falling back to the coarse RA/Dec solution. "
                "n_obj=%d n_enabled=%d n_close=%d nearest_mm[min/med/max]=%.3f/%.3f/%.3f "
                "coarse_ra_arcsec=%.6f coarse_dec_arcsec=%.6f enabled_camera_ids=%s",
                n_obj,
                n_enabled,
                n_close,
                close_min,
                close_median,
                close_max,
                coarse_ra_coeff * d_ra * 3600.0,
                coarse_dec_coeff * d_de * 3600.0,
                enabled_camera_ids,
            )
            lstsq_coeffs = self._build_coefficients(coarse_ra_coeff, coarse_dec_coeff, fit_inr, fit_scale)
            residual = np.empty((0, 1))
            rank = 0
            sv = np.empty(0)
        else:
            newbasis = basis[np.concatenate([fit_mask, fit_mask])]
            newerr   = err[np.concatenate([fit_mask, fit_mask])]
            lstsq_coeffs, residual, rank, sv = np.linalg.lstsq(newbasis, newerr, rcond = None)

        match_obj_xy = np.stack([match_obj_xdp,match_obj_ydp]).transpose()
        match_cat_xy = np.stack([match_cat_xdp,match_cat_ydp]).transpose()
        err_xy       = np.stack([errx,erry]).transpose()
        resid_xy = (((err-np.dot(basis,lstsq_coeffs))[:,0]).reshape([2,-1])).transpose()

        # ── Phase 6: Iterative outlier rejection ──────────────────────────────
        # Up to 5 iterations: reject matches whose residual exceeds
        # min(3 × median_residual, maxresid), refit on the surviving inliers,
        # and update the threshold.  Loop terminates early once the threshold
        # stops changing (convergence).
        max_rejection_threshold = maxresid
        rejection_threshold = np.min(np.array([np.nanmedian(np.sqrt(np.sum(resid_xy**2,axis=1)))*3, max_rejection_threshold]))
        for rej_itr in range(5):
            resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))

            # Keep exact-zero residuals in the inlier set; otherwise a perfect
            # first fit can empty the refit and collapse the solution to zeros.
            inlier_mask = resid_r <= rejection_threshold
            # Restrict the refit and threshold update to enabled cameras only.
            # Disabled-camera residuals are predictions only and must not drive
            # the rejection threshold, or they could tighten it and reject valid
            # enabled-camera detections that the fit actually depends on.
            enabled_inlier_mask = inlier_mask & enabled_mask
            inlier_flat_mask = np.concatenate([enabled_inlier_mask, enabled_inlier_mask])

            if not np.any(inlier_flat_mask):
                logger.warning(
                    "RADECInRShiftA rejection step produced no inliers; retaining the previous solution and stopping. "
                    "iteration=%d rejection_threshold=%.6f n_obj=%d n_enabled=%d n_close=%d n_fit=%d",
                    rej_itr,
                    rejection_threshold,
                    n_obj,
                    n_enabled,
                    n_close,
                    n_fit,
                )
                break

            basis2 = basis[inlier_flat_mask]
            err2   = err[inlier_flat_mask]
            lstsq_coeffs, residual, rank, sv = np.linalg.lstsq(basis2, err2, rcond = None)
            resid_xy = (((err-np.dot(basis,lstsq_coeffs))[:,0]).reshape([2,-1])).transpose()
            rejection_threshold_old = rejection_threshold
            resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
            rejection_threshold = np.min(np.array([np.nanmedian(resid_r[enabled_inlier_mask])*3,max_rejection_threshold]))
            if(rejection_threshold == rejection_threshold_old):
                break

        resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
        vcx = np.array([resid_r<=rejection_threshold]).transpose()
        camera_enabled = np.array([enabled_mask], dtype=float).transpose()
        match_result = np.block([match_obj_xy, match_cat_xy, err_xy, resid_xy, vcx, min_dist_index.reshape(-1,1), camera_enabled])

        # ── Phase 7: Unit conversion ──────────────────────────────────────────
        # Multiply the dimensionless least-squares coefficients by the
        # corresponding perturbation sizes to recover physical offsets.
        ra_offset    = 0.0
        de_offset    = 0.0
        inr_offset   = np.nan
        scale_offset = np.nan

        if fit_inr == 1 and fit_scale == 1:
            ra_offset    = lstsq_coeffs[0][0] * d_ra
            de_offset    = lstsq_coeffs[1][0] * d_de
            inr_offset   = lstsq_coeffs[2][0] * d_inr
            scale_offset = lstsq_coeffs[3][0] * d_scl
        elif fit_inr == 1 and fit_scale == 0:
            ra_offset    = lstsq_coeffs[0][0] * d_ra
            de_offset    = lstsq_coeffs[1][0] * d_de
            inr_offset   = lstsq_coeffs[2][0] * d_inr
        elif fit_inr == 0 and fit_scale == 1:
            ra_offset    = lstsq_coeffs[0][0] * d_ra
            de_offset    = lstsq_coeffs[1][0] * d_de
            scale_offset = lstsq_coeffs[2][0] * d_scl
        else:
            ra_offset    = lstsq_coeffs[0][0] * d_ra
            de_offset    = lstsq_coeffs[1][0] * d_de

        return ra_offset, de_offset, inr_offset, scale_offset, match_result

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        """Compute the catalog basis arrays for both detector halves.

        This method provides a stable public name for the basis computation;
        it is a pure delegate to :meth:`makeBasisPfi`.  All arguments and
        return values are identical — refer to :meth:`makeBasisPfi` for full
        documentation.
        """
        v_0,v_1 = PFS.makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl)
        return v_0,v_1

    def makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        """Compute detector-plane positions and RA/Dec/InR Jacobians for
        catalog stars in both detector halves.

        The focal-plane coordinate ``celestial2focalplane`` is evaluated
        twice per star — once with ``z`` (zeros, i.e. the LEFT-half flag
        set to 0) and once with ``o`` (ones, i.e. the RIGHT-half flag set
        to 1).  This flag selects which half of the AG detector the model
        uses; the two results are returned as separate basis arrays so that
        ``RADECInRShiftA`` can choose the geometrically appropriate
        Jacobian for each detected source.

        ``sep`` is the angular separation [deg] between the telescope
        pointing and each guide star on the sky.
        ``zpa`` is the corresponding zenith position angle [deg].

        The returned arrays ``v_0`` and ``v_1`` have shape (M, 8), where
        M is the number of catalog stars and the 8 columns are:
          col 0: xdp  — detector-plane x [mm]
          col 1: ydp  — detector-plane y [mm]
          col 2: dx/dRA   — ∂x_dp/∂(RA perturbation)
          col 3: dy/dRA   — ∂y_dp/∂(RA perturbation)
          col 4: dx/dDec  — ∂x_dp/∂(Dec perturbation)
          col 5: dy/dDec  — ∂y_dp/∂(Dec perturbation)
          col 6: dx/dInR  — ∂x_dp/∂(InR perturbation)
          col 7: dy/dInR  — ∂y_dp/∂(InR perturbation)

        The magnitude column is intentionally absent from the return value;
        it is inserted by the caller (``FieldAcquisitionAndFocusing.py``)
        via ``np.insert(..., 2, magnitude_values, axis=1)`` so that the
        basis computation itself remains magnitude-agnostic.

        Parameters
        ----------
        tel_ra : float or np.ndarray
            Telescope pointing right ascension [deg].
        tel_de : float or np.ndarray
            Telescope pointing declination [deg].
        str_ra : np.ndarray
            Guide-star catalog right ascensions [deg].
        str_de : np.ndarray
            Guide-star catalog declinations [deg].
        t : astropy.time.Time or equivalent
            Observation epoch (used for refraction and aberration).
        adc : float
            ADC (atmospheric dispersion corrector) angle [deg].
        inr : float
            Instrument rotation angle [deg].
        m2pos3 : float
            Secondary mirror piston position along the optical axis [mm].
        wl : float or np.ndarray
            Wavelength [nm] for chromatic corrections.

        Returns
        -------
        v_0 : np.ndarray, shape (M, 8)
            Basis array for the LEFT detector half (``celestial2focalplane``
            called with flag = 0).
        v_1 : np.ndarray, shape (M, 8)
            Basis array for the RIGHT detector half (``celestial2focalplane``
            called with flag = 1).
        """
        sep0,zpa0 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de,      str_ra, str_de, wl, t)
        sep1,zpa1 = pfs.Subaru.starSepZPA(self, tel_ra+d_ra, tel_de,      str_ra, str_de, wl, t)
        sep2,zpa2 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de+d_de, str_ra, str_de, wl, t)

        az,el = pfs.Subaru.radec2azel(self, tel_ra, tel_de, wl, t)

        # z = zeros flag → LEFT detector half; o = ones flag → RIGHT detector half
        z = np.zeros_like(sep0)
        o = np.ones_like(sep0)

        xfp0_0,yfp0_0 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,inr,el,m2pos3,wl,z)
        xfp1_0,yfp1_0 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,inr,el,m2pos3,wl,z)
        xfp2_0,yfp2_0 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,inr,el,m2pos3,wl,z)

        xfp0_1,yfp0_1 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,inr,el,m2pos3,wl,o)
        xfp1_1,yfp1_1 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,inr,el,m2pos3,wl,o)
        xfp2_1,yfp2_1 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,inr,el,m2pos3,wl,o)

        xfp0 = 0.5*(xfp0_0+xfp0_1)
        xfp1 = 0.5*(xfp1_0+xfp1_1)
        xfp2 = 0.5*(xfp2_0+xfp2_1)

        yfp0 = 0.5*(yfp0_0+yfp0_1)
        yfp1 = 0.5*(yfp1_0+yfp1_1)
        yfp2 = 0.5*(yfp2_0+yfp2_1)

        xdp0,ydp0 = pfs.PFS.fp2pfi(self, xfp0,yfp0,inr)
        xdp1,ydp1 = pfs.PFS.fp2pfi(self, xfp1,yfp1,inr)
        xdp2,ydp2 = pfs.PFS.fp2pfi(self, xfp2,yfp2,inr)
        xdp3,ydp3 = pfs.PFS.fp2pfi(self, xfp0,yfp0,inr+d_inr)

        xdp0_0,ydp0_0 = pfs.PFS.fp2pfi(self, xfp0_0,yfp0_0,inr)
        xdp1_0,ydp1_0 = pfs.PFS.fp2pfi(self, xfp1_0,yfp1_0,inr)
        xdp2_0,ydp2_0 = pfs.PFS.fp2pfi(self, xfp2_0,yfp2_0,inr)
        xdp3_0,ydp3_0 = pfs.PFS.fp2pfi(self, xfp0_0,yfp0_0,inr+d_inr)

        xdp0_1,ydp0_1 = pfs.PFS.fp2pfi(self, xfp0_1,yfp0_1,inr)
        xdp1_1,ydp1_1 = pfs.PFS.fp2pfi(self, xfp1_1,yfp1_1,inr)
        xdp2_1,ydp2_1 = pfs.PFS.fp2pfi(self, xfp2_1,yfp2_1,inr)
        xdp3_1,ydp3_1 = pfs.PFS.fp2pfi(self, xfp0_1,yfp0_1,inr+d_inr)

        dxdpdra_0 = xdp1_0-xdp0_0
        dydpdra_0 = ydp1_0-ydp0_0
        dxdpdde_0 = xdp2_0-xdp0_0
        dydpdde_0 = ydp2_0-ydp0_0
        dxdpdinr_0= xdp3_0-xdp0_0
        dydpdinr_0= ydp3_0-ydp0_0

        dxdpdra_1 = xdp1_1-xdp0_1
        dydpdra_1 = ydp1_1-ydp0_1
        dxdpdde_1 = xdp2_1-xdp0_1
        dydpdde_1 = ydp2_1-ydp0_1
        dxdpdinr_1= xdp3_1-xdp0_1
        dydpdinr_1= ydp3_1-ydp0_1

        v_0 = np.transpose(np.stack([xdp0_0,ydp0_0,dxdpdra_0,dydpdra_0,dxdpdde_0,dydpdde_0,dxdpdinr_0,dydpdinr_0]))
        v_1 = np.transpose(np.stack([xdp0_1,ydp0_1,dxdpdra_1,dydpdra_1,dxdpdde_1,dydpdde_1,dxdpdinr_1,dydpdinr_1]))

        return v_0,v_1

    def agarray2momentdifference(self, array, maxellip, maxsize, minsize):
        """Compute the per-CCD second-moment difference as a focus proxy.

        The spider vanes cast shadows on the focal-plane illumination pattern.
        When the telescope is out of focus the spider shadow shifts the PSF
        centroid differently for sources with (``flag == 1``) and without
        (``flag == 0``) the spider in their light path.  The difference in
        averaged second moments between these two populations therefore
        encodes the sign and magnitude of the focus error.

        This method assumes exactly 6 AG cameras (CCD IDs 1–6).

        Parameters
        ----------
        array : np.ndarray, shape (N, 8)
            Raw AG source array. Columns follow the same layout as the
            ``agarray`` parameter of :meth:`sourceFilter`:
              0: ccd        — CCD index (1–6)
              1: id         — source ID
              2: xc [mm]    — centroid x
              3: yc [mm]    — centroid y
              4: flux       — integrated flux [counts]
              5: semi-major [px]
              6: semi-minor [px]
              7: flag       — spider flag: 0 = without spider, 1 = with spider
        maxellip : float
            Passed to :meth:`sourceFilter` (see that method for details).
        maxsize : float
            Passed to :meth:`sourceFilter`.
        minsize : float
            Passed to :meth:`sourceFilter`.

        Returns
        -------
        moment_diff_per_ccd : np.ndarray, shape (6,)
            Per-CCD moment difference
            ``median(a² + b²)_without_spider − median(a² + b²)_with_spider``
            [pixels²], where ``a`` and ``b`` are the semi-major and
            semi-minor axes of the PSF.  A positive value indicates the PSF
            is broader without the spider, which corresponds to a specific
            sign of defocus.  Entries are ``nan`` if a CCD has too few
            detections to compute a median.
        """
        ##### array
        ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix] Flag[0 or 1]
        filtered_agarray, valid_mask = PFS.sourceFilter(self, array, maxellip, maxsize, minsize)
        moment_diff_per_ccd=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for ccdid in range(1,7):
            ccd_array = filtered_agarray[np.where(filtered_agarray[:,0]==ccdid)]
            array_wosp = ccd_array[np.where(ccd_array[:,7]==0)]
            array_wisp = ccd_array[np.where(ccd_array[:,7]==1)]

            moment_wosp = np.median((array_wosp[:,5]**2+array_wosp[:,6]**2))
            moment_wisp = np.median((array_wisp[:,5]**2+array_wisp[:,6]**2))

            moment_diff_per_ccd[ccdid-1]=moment_wosp-moment_wisp

        return moment_diff_per_ccd

    def momentdifference2focuserror(self, momentdifference):
        """Convert a per-CCD second-moment difference to a focus error.

        Uses an empirical linear calibration derived from lab/on-sky
        measurements:
          ``focus_error [mm] = momentdifference [px²] × 0.0086 − 0.026``

        Calibration constants:
          - ``0.0086`` [mm px⁻²]: slope — relates PSF moment spread to
            physical defocus along the optical axis.
          - ``0.026``  [mm]:       intercept — accounts for a systematic
            offset between measured moment difference and true focus zero.

        Sign convention: a positive focus error means the focal plane is
        displaced in the positive M2 piston direction (toward the primary).

        Parameters
        ----------
        momentdifference : float or np.ndarray
            Second-moment difference (without-spider minus with-spider)
            [pixels²] as returned by :meth:`agarray2momentdifference`.

        Returns
        -------
        focuserror : float or np.ndarray
            Estimated focus error [mm].
        """
        # momentdifference [pixel^2]
        # focuserror [mm]
        focuserror = momentdifference * 0.0086 - 0.026

        return focuserror
