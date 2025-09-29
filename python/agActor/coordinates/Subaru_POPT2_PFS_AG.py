import logging

import numpy as np
from pfs.utils.coordinates import Subaru_POPT2_PFS as pfs
from pfs.utils.datamodel.ag import SourceDetectionFlag

logger = logging.getLogger(__name__)

### perturbation
d_ra  = 1.0/3600.0
d_de  = 1.0/3600.0
d_inr = 0.01
d_scl = 1.0e-05


class PFS():
    def sourceFilter(self, detected_objects, maxellip, maxsize, minsize):
        bad_ellip = ~detected_objects.eval('ellipticity < @maxellip')
        bad_size = ~detected_objects.eval('@minsize < size < @maxsize')

        detected_objects.loc[bad_ellip, 'flags'] |= SourceDetectionFlag.BAD_ELLIP.value
        detected_objects.loc[bad_size, 'flags'] |= SourceDetectionFlag.BAD_SHAPE.value

        return detected_objects

    def RADECInRShiftA(self, obj_xdp, obj_ydp, obj_flag, v0, v1, inrflag, scaleflag, maxresid=0.5):
        """
        Estimate small pointing and instrument offsets by matching observed AG detections
        to model-predicted catalog positions, then solving a linearized system.

        What it does (high-level):
        - Builds a linear basis of partial derivatives that describe how detector (x, y)
          shift with respect to changes in RA, Dec, instrument rotator angle (INR) and
          a small plate-scale term.
        - Finds nearest catalog predictions to each observed object (with optional
          condition/flag selecting between two catalog states v0 and v1, e.g., without/with glass).
        - Computes a coarse RA/Dec translation using the local derivatives.
        - Refines the match and solves a least-squares problem to estimate the offsets.
        - Iteratively rejects outliers based on residuals and returns final offsets and a
          detailed matching/residuals report.

        Parameters
        ----------
        obj_xdp : array-like, shape (N,)
            Observed detector x positions [mm] of AG objects.
        obj_ydp : array-like, shape (N,)
            Observed detector y positions [mm] of AG objects.
        obj_flag : array-like, shape (N,)
            Per-object flag; objects with value 1.0 use the v1 catalog/basis, others use v0.
        v0 : ndarray, shape (M, >=9)
            Catalog/basis for condition 0. Columns are:
              [x_cat, y_cat, mag, dx/dRA, dy/dRA, dx/dDec, dy/dDec, dx/dINR, dy/dINR].
            A small scale basis is synthesized as [x_cat, y_cat] * d_scl.
        v1 : ndarray, shape (M, >=9)
            Catalog/basis for condition 1 (same column convention as v0).
        inrflag : int {0,1}
            If 1, include INR derivative in the fit; if 0, omit it.
        scaleflag : int {0,1}
            If 1, include scale derivative in the fit; if 0, omit it.
        maxresid : float, optional
            Upper bound [mm] for the residual threshold used in outlier rejection.

        Returns
        -------
        ra_offset : float
            Estimated RA pointing offset in sky units, scaled by d_ra (arcsec converted to model step).
        de_offset : float
            Estimated Dec pointing offset in sky units, scaled by d_de.
        inr_offset : float or NaN
            Estimated instrument rotator offset (only meaningful if inrflag==1), scaled by d_inr.
        scale_offset : float or NaN
            Estimated small scale term (only meaningful if scaleflag==1), scaled by d_scl.
        mr : ndarray, shape (N, 9)
            Match/report matrix with columns:
              [obj_x, obj_y, cat_x, cat_y, err_x, err_y, resid_x, resid_y, valid_mask, match_index]
            where valid_mask is 1 for accepted matches after outlier rejection.
        """
        # Unpack catalog basis arrays for two different conditions (e.g., with/without glass)
        cat_xdp_0 = v0[:,0]
        cat_ydp_0 = v0[:,1]
        cat_mag_0 = v0[:,2]
        dxra_0    = v0[:,3]
        dyra_0    = v0[:,4]
        dxde_0    = v0[:,5]
        dyde_0    = v0[:,6]
        dxinr_0   = v0[:,7]
        dyinr_0   = v0[:,8]
        dxscl_0   = v0[:,0]*d_scl
        dyscl_0   = v0[:,1]*d_scl

        cat_xdp_1 = v1[:,0]
        cat_ydp_1 = v1[:,1]
        cat_mag_1 = v1[:,2]
        dxra_1    = v1[:,3]
        dyra_1    = v1[:,4]
        dxde_1    = v1[:,5]
        dyde_1    = v1[:,6]
        dxinr_1   = v1[:,7]
        dyinr_1   = v1[:,8]
        dxscl_1   = v1[:,0]*d_scl
        dyscl_1   = v1[:,1]*d_scl

        # Average the basis vectors between the two conditions
        dxra  = (dxra_0  + dxra_1 )/2.0
        dyra  = (dyra_0  + dyra_1 )/2.0
        dxde  = (dxde_0  + dxde_1 )/2.0
        dyde  = (dyde_0  + dyde_1 )/2.0
        dxinr = (dxinr_0 + dxinr_1)/2.0
        dyinr = (dyinr_0 + dyinr_1)/2.0
        dxscl = (dxscl_0 + dxscl_1)/2.0
        dyscl = (dyscl_0 + dyscl_1)/2.0

        # Identify objects with flag==1.0 (e.g., with glass)
        flg = np.where(obj_flag==1.0)

        n_obj = (obj_xdp.shape)[0]

        # Compute differences between object and catalog positions for both conditions
        xdiff_0 = np.transpose([obj_xdp])-cat_xdp_0
        ydiff_0 = np.transpose([obj_ydp])-cat_ydp_0
        xdiff_1 = np.transpose([obj_xdp])-cat_xdp_1
        ydiff_1 = np.transpose([obj_ydp])-cat_ydp_1

        # Use appropriate differences depending on flag
        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]

        # Find closest catalog match for each object
        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj), dtype='int'),min_dist_index

        # Estimate RA/DEC offsets using median of matched differences
        rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

        # Apply estimated offsets to catalog positions and recompute differences
        xdiff_0 = np.transpose([obj_xdp])-(cat_xdp_0+rCRA*dxra+rCDE*dxde)
        ydiff_0 = np.transpose([obj_ydp])-(cat_ydp_0+rCRA*dyra+rCDE*dyde)
        xdiff_1 = np.transpose([obj_xdp])-(cat_xdp_1+rCRA*dxra+rCDE*dxde)
        ydiff_1 = np.transpose([obj_ydp])-(cat_ydp_1+rCRA*dyra+rCDE*dyde)

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]

        # Find closest catalog match again after offset correction
        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj), dtype='int'),min_dist_index

        # Filter matches by distance threshold
        f  = dist[min_dist_indices] < 2.0

        # Prepare matched object and catalog arrays
        match_obj_xdp  = obj_xdp
        match_obj_ydp  = obj_ydp
        match_obj_flag = obj_flag

        # Gather matched catalog basis vectors for both conditions
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

        # Use appropriate catalog values depending on flag
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

        flg = np.where(match_obj_flag==1.0)
        match_cat_xdp[flg] = match_cat_xdp_1[flg]
        match_cat_ydp[flg] = match_cat_ydp_1[flg]
        match_cat_mag[flg] = match_cat_mag_1[flg]
        match_dxra[flg]    = match_dxra_1[flg]
        match_dyra[flg]    = match_dyra_1[flg]
        match_dxde[flg]    = match_dxde_1[flg]
        match_dyde[flg]    = match_dyde_1[flg]
        match_dxinr[flg]   = match_dxinr_1[flg]
        match_dyinr[flg]   = match_dyinr_1[flg]
        match_dxscl[flg]   = match_dxscl_1[flg]
        match_dyscl[flg]   = match_dyscl_1[flg]

        # Build basis matrix for least squares fit
        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])
        dscl = np.concatenate([match_dxscl,match_dyscl])

        # Select which basis vectors to use depending on flags
        if inrflag == 1 and scaleflag == 1:
            basis= np.stack([dra,dde,dinr,dscl]).transpose()
        elif inrflag == 1 and scaleflag == 0:
            basis= np.stack([dra,dde,dinr]).transpose()
        elif inrflag == 0 and scaleflag == 1:
            basis= np.stack([dra,dde,dscl]).transpose()
        else:
            basis= np.stack([dra,dde]).transpose()

        # Compute error vector between matched object and catalog positions
        errx = match_obj_xdp - match_cat_xdp
        erry = match_obj_ydp - match_cat_ydp
        err  = np.array([np.concatenate([errx,erry])]).transpose()

        # Filter basis and error arrays by match quality
        newbasis = basis[np.concatenate([f,f])]
        newerr   = err[np.concatenate([f,f])]

        # Solve for offsets using least squares
        A, residual, rank, sv = np.linalg.lstsq(newbasis, newerr, rcond = None)

        # Compute residuals for each match
        match_obj_xy = np.stack([match_obj_xdp,match_obj_ydp]).transpose()
        match_cat_xy = np.stack([match_cat_xdp,match_cat_ydp]).transpose()
        err_xy       = np.stack([errx,erry]).transpose()
        resid_xy = (((err-np.dot(basis,A))[:,0]).reshape([2,-1])).transpose()

        #### Outlier rejection loop (removes matches with large residuals)
        rej_thres_lim = maxresid
        rej_thres = np.min(np.array([np.nanmedian(np.sqrt(np.sum(resid_xy**2,axis=1)))*3, rej_thres_lim]))
        logger.info(f'Rejection threshold: {rej_thres} mm (limit {rej_thres_lim} mm)')
        for rej_itr in range(5):
            resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
            vc  = np.where(np.concatenate([resid_r, resid_r], 0) < rej_thres)
            vch = np.where(np.concatenate([resid_r], 0) < rej_thres)
            basis2 = basis[vc]
            err2   = err[vc]
            A, residual, rank, sv = np.linalg.lstsq(basis2, err2, rcond = None)
            resid_xy = (((err-np.dot(basis,A))[:,0]).reshape([2,-1])).transpose()
            rej_thres_old = rej_thres
            resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
            rej_thres = np.min(np.array([np.nanmedian(resid_r[vch])*3,rej_thres_lim]))
            logger.info(f'Rejection threshold: {rej_thres} mm (limit {rej_thres_lim} mm)')
            if(rej_thres == rej_thres_old):
                break

        # Final residuals and mask for accepted matches
        logger.info(f'Final Rejection threshold: {rej_thres} mm (limit {rej_thres_lim} mm)')
        resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
        vcx = np.array([resid_r<rej_thres]).transpose()
        mr = np.block([match_obj_xy, match_cat_xy, err_xy, resid_xy, vcx, min_dist_index.reshape(-1,1)])

        # Convert fit coefficients to physical offsets
        ra_offset    = 0.0
        de_offset    = 0.0
        inr_offset   = np.nan
        scale_offset = np.nan

        if inrflag == 1 and scaleflag == 1:
            ra_offset    = A[0][0] * d_ra
            de_offset    = A[1][0] * d_de
            inr_offset   = A[2][0] * d_inr
            scale_offset = A[3][0] * d_scl
        elif inrflag == 1 and scaleflag == 0:
            ra_offset    = A[0][0] * d_ra
            de_offset    = A[1][0] * d_de
            inr_offset   = A[2][0] * d_inr
        elif inrflag == 0 and scaleflag == 1:
            ra_offset    = A[0][0] * d_ra
            de_offset    = A[1][0] * d_de
            scale_offset = A[2][0] * d_scl
        else:
            ra_offset    = A[0][0] * d_ra
            de_offset    = A[1][0] * d_de

        # Return offsets and match results
        return ra_offset, de_offset, inr_offset, scale_offset, mr

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        v_0,v_1 = PFS.makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl)
        return v_0,v_1

    def makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        sep0,zpa0 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de,      str_ra, str_de, wl, t)
        sep1,zpa1 = pfs.Subaru.starSepZPA(self, tel_ra+d_ra, tel_de,      str_ra, str_de, wl, t)
        sep2,zpa2 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de+d_de, str_ra, str_de, wl, t)

        az,el = pfs.Subaru.radec2azel(self, tel_ra, tel_de, wl, t)

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

        dxdpdra = xdp1-xdp0
        dydpdra = ydp1-ydp0
        dxdpdde = xdp2-xdp0
        dydpdde = ydp2-ydp0
        dxdpdinr= xdp3-xdp0
        dydpdinr= ydp3-ydp0

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

        v_a = np.transpose(np.stack([xdp0,ydp0,dxdpdra,dydpdra,dxdpdde,dydpdde,dxdpdinr,dydpdinr]))
        v_0 = np.transpose(np.stack([xdp0_0,ydp0_0,dxdpdra_0,dydpdra_0,dxdpdde_0,dydpdde_0,dxdpdinr_0,dydpdinr_0]))
        v_1 = np.transpose(np.stack([xdp0_1,ydp0_1,dxdpdra_1,dydpdra_1,dxdpdde_1,dydpdde_1,dxdpdinr_1,dydpdinr_1]))

        # return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr
        return v_0,v_1

    def agarray2momentdifference(self, filtered_agarray, maxellip, maxsize, minsize):
        ##### array
        ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix] Flag[0 or 1]
        # filtered_agarray, v = PFS.sourceFilter(self, array, maxellip, maxsize, minsize)
        outarray=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for ccdid in range(1,7):
            array = filtered_agarray[np.where(filtered_agarray[:,0]==ccdid)]
            array_wosp = array[np.where(array[:,7]==0)]
            array_wisp = array[np.where(array[:,7]==1)]

            moment_wosp = np.median((array_wosp[:,5]**2+array_wosp[:,6]**2))
            moment_wisp = np.median((array_wisp[:,5]**2+array_wisp[:,6]**2))

            outarray[ccdid-1]=moment_wosp-moment_wisp

        return outarray

    def momentdifference2focuserror(self, momentdifference):
        # momentdifference [pixel^2]
        # focuserror [mm]
        focuserror = momentdifference * 0.0086 - 0.026

        return focuserror
