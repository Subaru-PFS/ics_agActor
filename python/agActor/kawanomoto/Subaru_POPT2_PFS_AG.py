# -*- coding: utf-8 -*-
import numpy as np
import sys

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

from astropy.utils import iers
iers.conf.auto_download = True

if __name__ == '__main__':
    import Subaru_POPT2_PFS as pfs
else:
    from . import Subaru_POPT2_PFS as pfs

###
pfs.inr_zero_offset = -90

### perturbation
d_ra  = 1.0/3600.0
d_de  = 1.0/3600.0
d_inr = 0.01

### pfi parity (flip y)
pfi_parity = -1.0 # -1 or +1, 

# ### constants proper to PFS camera
# pfs_inr_zero_offset        =  0.00 # in degree
# pfs_detector_zero_offset_x =  0.00 # in mm
# pfs_detector_zero_offset_y =  0.00 # in mm

# ### Subaru location
# sbr_lat =   +19.8225
# sbr_lon =  +204.523972222
# sbr_hei = +4163.0
# Lsbr = ac.EarthLocation(lat=sbr_lat,lon=sbr_lon,height=sbr_hei)

# ### misc.
# sbr_press  = 620.0

# class Subaru():
#     def radec2inr(self, tel_ra, tel_de, t):
#         pr  = 0.0   # Subaru InR ignore atmospheric refraction
#         wl  = 0.62  # so wavelength is selected freely in visible light
 
#         tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5',equinox='J2000.0')
#         np_coord  = ac.SkyCoord(ra=0.0,    dec=90.0,   unit=(au.deg, au.deg), frame='fk5',equinox=t)
#         frame_subaru = ac.AltAz(obstime  = t, location = Lsbr, \
#                                 pressure = pr*au.hPa, obswl = wl*au.micron)
#         tel_altaz = tel_coord.transform_to(frame_subaru)
#         np_altaz  = np_coord.transform_to(frame_subaru)
#         inr_cal = (tel_altaz.position_angle(np_altaz).degree-180)%360-180
#         return inr_cal

#     def starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t):
#         tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5')
#         str_coord = ac.SkyCoord(ra=str_ra, dec=str_de, unit=(au.deg, au.deg), frame='fk5')
#         frame_subaru = ac.AltAz(obstime  = t, location = Lsbr,\
#                                 pressure = sbr_press*au.hPa, obswl = wl*au.micron)
#         tel_altaz = tel_coord.transform_to(frame_subaru)
#         str_altaz = str_coord.transform_to(frame_subaru)

#         str_sep =  tel_altaz.separation(str_altaz).degree
#         str_zpa = -tel_altaz.position_angle(str_altaz).degree
#         return str_sep, str_zpa

#     def starRADEC(self, tel_ra, tel_de, str_sep, str_zpa, wl, t):
#         str_sep = str_sep*au.degree
#         str_zpa = str_zpa*au.degree

#         tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5')
#         frame_subaru = ac.AltAz(obstime  = t, location = Lsbr,\
#                                 pressure = sbr_press*au.hPa, obswl = wl*au.micron)
#         tel_altaz = tel_coord.transform_to(frame_subaru)
#         str_altaz = tel_altaz.directional_offset_by(str_zpa, str_sep)
#         str_coord = str_altaz.transform_to('fk5')
#         ra = str_coord.ra.degree
#         de = str_coord.dec.degree
#         return ra,de

#     def radec2radecplxpm(self, gaia_epoch, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t):
#         str_plx[np.where(str_plx<0.00001)]=0.00001
#         str_coord = ac.SkyCoord(ra=str_ra, dec=str_de, unit=(au.deg, au.deg),
#                                 distance=ac.Distance(parallax=str_plx * au.mas, allow_negative=False),             
#                                 pm_ra_cosdec = str_pmRA * au.mas/au.yr,
#                                 pm_dec = str_pmDE * au.mas/au.yr,
#                                 obstime=Time(gaia_epoch, format='decimalyear'),
#                                 frame='fk5')
#         str_coord_obstime = str_coord.apply_space_motion(at.Time(t))
#         ra = str_coord_obstime.ra.degree
#         de = str_coord_obstime.dec.degree
#         return ra,de

class PFS():
    def sourceFilter(self, agarray, maxellip, maxsize, minsize):
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

        # flag condition (0: no glass, 1: with glass, 2-7 : edge and/or satulate)
        cflag  = ag_flag < 2
        
        v = cellip*csizeU*csizeL*cflag
        
        vdatanum = np.sum(v)

        oarray = np.zeros((vdatanum,8))
        oarray[:,0] = ag_ccd[v]
        oarray[:,1] = ag_id[v]
        oarray[:,2] = ag_xc[v]
        oarray[:,3] = ag_yc[v]
        oarray[:,4] = ag_flx[v]
        oarray[:,5] = ag_smma[v]
        oarray[:,6] = ag_smmi[v]
        oarray[:,7] = ag_flag[v]
        
        return oarray, v

    def RADECInRShift(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1):
        inrflag = 1
        ra_offset, de_offset, inr_offset, mr, *ex = PFS.RADECInRShiftA(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1, inrflag)
        rs = mr[:,6]**2+mr[:,7]**2
        rs[mr[:,8]==0.0]=np.nan
        md = np.nanmedian(np.sqrt(rs))
        return ra_offset, de_offset, inr_offset, mr, md, *ex
    
    def RADECShift(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1):
        inrflag = 0
        ra_offset, de_offset, inr_offset, mr, *ex = PFS.RADECInRShiftA(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1, inrflag)
        rs = mr[:,6]**2+mr[:,7]**2
        rs[mr[:,8]==0.0]=np.nan
        md = np.nanmedian(np.sqrt(rs))
        return ra_offset, de_offset, inr_offset, mr, md, *ex

    def RADECInRShiftA(self, obj_xdp, obj_ydp, obj_int, obj_flag, v0, v1, inrflag):
        cat_xdp_0 = v0[:,0]
        cat_ydp_0 = v0[:,1]
        cat_mag_0 = v0[:,2]
        dxra_0    = v0[:,3]
        dyra_0    = v0[:,4]
        dxde_0    = v0[:,5]
        dyde_0    = v0[:,6]
        dxinr_0   = v0[:,7]
        dyinr_0   = v0[:,8]
        
        cat_xdp_1 = v1[:,0]
        cat_ydp_1 = v1[:,1]
        cat_mag_1 = v1[:,2]
        dxra_1    = v1[:,3]
        dyra_1    = v1[:,4]
        dxde_1    = v1[:,5]
        dyde_1    = v1[:,6]
        dxinr_1   = v1[:,7]
        dyinr_1   = v1[:,8]

        dxra  = (dxra_0  + dxra_1 )/2.0
        dyra  = (dyra_0  + dyra_1 )/2.0
        dxde  = (dxde_0  + dxde_1 )/2.0
        dyde  = (dyde_0  + dyde_1 )/2.0
        dxinr = (dxinr_0 + dxinr_1)/2.0
        dyinr = (dyinr_0 + dyinr_1)/2.0

        flg = np.where(obj_flag==1.0)

        n_obj = (obj_xdp.shape)[0]
        xdiff_0 = np.transpose([obj_xdp])-cat_xdp_0
        ydiff_0 = np.transpose([obj_ydp])-cat_ydp_0
        xdiff_1 = np.transpose([obj_xdp])-cat_xdp_1
        ydiff_1 = np.transpose([obj_ydp])-cat_ydp_1

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]

        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index        
        rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

        xdiff_0 = np.transpose([obj_xdp])-(cat_xdp_0+rCRA*dxra+rCDE*dxde)
        ydiff_0 = np.transpose([obj_ydp])-(cat_ydp_0+rCRA*dyra+rCDE*dyde)

        xdiff_1 = np.transpose([obj_xdp])-(cat_xdp_1+rCRA*dxra+rCDE*dxde)
        ydiff_1 = np.transpose([obj_ydp])-(cat_ydp_1+rCRA*dyra+rCDE*dyde)

        xdiff = np.copy(xdiff_0)
        ydiff = np.copy(ydiff_0)
        xdiff[flg]=xdiff_1[flg]
        ydiff[flg]=ydiff_1[flg]
        
        dist  = np.sqrt(xdiff**2+ydiff**2)

        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index

        f = dist[min_dist_indices]<2

        # print(f)
        
        match_obj_xdp  = obj_xdp[f]
        match_obj_ydp  = obj_ydp[f]
        match_obj_int  = obj_int[f]
        match_obj_flag = obj_flag[f]

        match_cat_xdp_0 = (cat_xdp_0[min_dist_index])[f]
        match_cat_ydp_0 = (cat_ydp_0[min_dist_index])[f]
        match_cat_mag_0 = (cat_mag_0[min_dist_index])[f]
        match_dxra_0    = (dxra_0[min_dist_index])[f]
        match_dyra_0    = (dyra_0[min_dist_index])[f]
        match_dxde_0    = (dxde_0[min_dist_index])[f]
        match_dyde_0    = (dyde_0[min_dist_index])[f]
        match_dxinr_0   = (dxinr_0[min_dist_index])[f]
        match_dyinr_0   = (dyinr_0[min_dist_index])[f]

        match_cat_xdp_1 = (cat_xdp_1[min_dist_index])[f]
        match_cat_ydp_1 = (cat_ydp_1[min_dist_index])[f]
        match_cat_mag_1 = (cat_mag_1[min_dist_index])[f]
        match_dxra_1    = (dxra_1[min_dist_index])[f]
        match_dyra_1    = (dyra_1[min_dist_index])[f]
        match_dxde_1    = (dxde_1[min_dist_index])[f]
        match_dyde_1    = (dyde_1[min_dist_index])[f]
        match_dxinr_1   = (dxinr_1[min_dist_index])[f]
        match_dyinr_1   = (dyinr_1[min_dist_index])[f]

        match_cat_xdp = np.copy(match_cat_xdp_0)
        match_cat_ydp = np.copy(match_cat_ydp_0)
        match_cat_mag = np.copy(match_cat_mag_0)
        match_dxra    = np.copy(match_dxra_0)
        match_dyra    = np.copy(match_dyra_0)
        match_dxde    = np.copy(match_dxde_0)
        match_dyde    = np.copy(match_dyde_0)
        match_dxinr   = np.copy(match_dxinr_0)
        match_dyinr   = np.copy(match_dyinr_0)

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

        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])

        if inrflag == 1:
            basis= np.stack([dra,dde,dinr]).transpose()
        else:
            basis= np.stack([dra,dde]).transpose()

        errx = match_obj_xdp - match_cat_xdp 
        erry = match_obj_ydp - match_cat_ydp 
        err  = np.array([np.concatenate([errx,erry])]).transpose()

        A, residual, rank, sv = np.linalg.lstsq(basis, err, rcond = None)

        match_obj_xy = np.stack([match_obj_xdp,match_obj_ydp]).transpose()
        match_cat_xy = np.stack([match_cat_xdp,match_cat_ydp]).transpose()
        err_xy       = np.stack([errx,erry]).transpose()
        resid_xy = (((err-np.dot(basis,A))[:,0]).reshape([2,-1])).transpose()

        #### one iteration to remove dr >= 0.5mm data
        resid_r = np.sqrt(np.sum(resid_xy**2,axis=1))
        vc  = np.where(np.concatenate([resid_r, resid_r], 0) < 0.5)
        # vcc = np.where(resid_r<0.5)
        vcx = np.array([resid_r<0.5]).transpose()
        basis2 = basis[vc]
        err2   = err[vc]
        A, residual, rank, sv = np.linalg.lstsq(basis2, err2, rcond = None)
        # match_obj_xy = np.stack([match_obj_xdp,match_obj_ydp]).transpose()[vcc]
        # match_cat_xy = np.stack([match_cat_xdp,match_cat_ydp]).transpose()[vcc]
        # errx = match_obj_xdp[vcc] - match_cat_xdp[vcc]
        # erry = match_obj_ydp[vcc] - match_cat_ydp[vcc]
        # err  = np.array([np.concatenate([errx,erry])]).transpose()
        # err_xy = np.stack([errx,erry]).transpose()
        # resid_xy = (((err2-np.dot(basis2,A))[:,0]).reshape([2,-1])).transpose()
        resid_xy = (((err-np.dot(basis,A))[:,0]).reshape([2,-1])).transpose()
        ####

        mr = np.block([match_obj_xy, match_cat_xy, err_xy, resid_xy, vcx])
        
        ra_offset  = 0.0
        de_offset  = 0.0
        inr_offset = 0.0
        
        if inrflag == 1:
            ra_offset  = A[0][0] * d_ra
            de_offset  = A[1][0] * d_de
            inr_offset = A[2][0] * d_inr
        else:
            ra_offset  = A[0][0] * d_ra
            de_offset  = A[1][0] * d_de
        
        return ra_offset, de_offset, inr_offset, mr, min_dist_index[f], f

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        v_0,v_1 = PFS.makeBasisFp(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl)
        # v_0,v_1 = PFS.makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl)
        return v_0,v_1
    
    def makeBasisFp(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        sep0,zpa0 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de,      str_ra, str_de, wl, t)
        sep1,zpa1 = pfs.Subaru.starSepZPA(self, tel_ra+d_ra, tel_de,      str_ra, str_de, wl, t)
        sep2,zpa2 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de+d_de, str_ra, str_de, wl, t)

        z = np.zeros_like(sep0)
        o = np.ones_like(sep0)
        
        xfp0_0,yfp0_0 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,z)
        xfp1_0,yfp1_0 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,z)
        xfp2_0,yfp2_0 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,z)

        xfp0_1,yfp0_1 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,o)
        xfp1_1,yfp1_1 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,o)
        xfp2_1,yfp2_1 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,o)

        xfp0 = 0.5*(xfp0_0+xfp0_1)
        xfp1 = 0.5*(xfp1_0+xfp1_1)
        xfp2 = 0.5*(xfp2_0+xfp2_1)

        yfp0 = 0.5*(yfp0_0+yfp0_1)
        yfp1 = 0.5*(yfp1_0+yfp1_1)
        yfp2 = 0.5*(yfp2_0+yfp2_1)

        xdp0,ydp0 = pfs.PFS.fp2dp(self, xfp0,yfp0,inr)
        xdp1,ydp1 = pfs.PFS.fp2dp(self, xfp1,yfp1,inr)
        xdp2,ydp2 = pfs.PFS.fp2dp(self, xfp2,yfp2,inr)
        xdp3,ydp3 = pfs.PFS.fp2dp(self, xfp0,yfp0,inr+d_inr)

        xdp0_0,ydp0_0 = pfs.PFS.fp2dp(self, xfp0_0,yfp0_0,inr)
        xdp1_0,ydp1_0 = pfs.PFS.fp2dp(self, xfp1_0,yfp1_0,inr)
        xdp2_0,ydp2_0 = pfs.PFS.fp2dp(self, xfp2_0,yfp2_0,inr)
        xdp3_0,ydp3_0 = pfs.PFS.fp2dp(self, xfp0_0,yfp0_0,inr+d_inr)

        xdp0_1,ydp0_1 = pfs.PFS.fp2dp(self, xfp0_1,yfp0_1,inr)
        xdp1_1,ydp1_1 = pfs.PFS.fp2dp(self, xfp1_1,yfp1_1,inr)
        xdp2_1,ydp2_1 = pfs.PFS.fp2dp(self, xfp2_1,yfp2_1,inr)
        xdp3_1,ydp3_1 = pfs.PFS.fp2dp(self, xfp0_1,yfp0_1,inr+d_inr)

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

    def makeBasisPfi(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        sep0,zpa0 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de,      str_ra, str_de, wl, t)
        sep1,zpa1 = pfs.Subaru.starSepZPA(self, tel_ra+d_ra, tel_de,      str_ra, str_de, wl, t)
        sep2,zpa2 = pfs.Subaru.starSepZPA(self, tel_ra,      tel_de+d_de, str_ra, str_de, wl, t)

        z = np.zeros_like(sep0)
        o = np.ones_like(sep0)
        
        xfp0_0,yfp0_0 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,z)
        xfp1_0,yfp1_0 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,z)
        xfp2_0,yfp2_0 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,z)

        xfp0_1,yfp0_1 = pfs.POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,o)
        xfp1_1,yfp1_1 = pfs.POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,o)
        xfp2_1,yfp2_1 = pfs.POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,o)

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

    def agpixel2pfipos(self, dsc):
        ccdpxsz  = 0.013
        # ccdcentt = np.array([ 0.0, -60.0,-120.0, 180.0, 120.0, 60.0])*np.pi/180.0
        ccdcentr = 241.314

        ccdoffx = np.array([-0.668, +0.081, +0.180, +0.357, +0.138, -0.077])
        ccdoffy = np.array([+0.405, +0.055, +0.357, -0.270, -0.444, -0.067])
        ccdoffx = ccdoffx - np.array([+0.015, +0.025, +0.014, +0.002, +0.022, +0.004]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdoffy = ccdoffy - np.array([-0.013, +0.012, +0.030, +0.013, -0.015, -0.024]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdrot  = np.array([-0.253368, +0.234505, +0.329449, +0.416894, +0.0589071, +0.234977])*np.pi/180.0

        ccdid  = dsc[:,0].astype(int)
        spotid = dsc[:,1].astype(int)
        flux   = dsc[:,2]
        xpix   = dsc[:,3]
        ypix   = dsc[:,4]
        ag_xy  = dsc[:,5]
        ag_xx  = dsc[:,6]
        ag_yy  = dsc[:,7]
        flag   = dsc[:,12]

        smma   = np.sqrt(0.5*(ag_xx+ag_yy)+np.sqrt((0.5*(ag_xx-ag_yy))**2+ag_xy**2))
        smmi   = np.sqrt(0.5*(ag_xx+ag_yy)-np.sqrt((0.5*(ag_xx-ag_yy))**2+ag_xy**2))

        xpixrot = np.zeros_like(xpix)
        ypixrot = np.zeros_like(ypix)
        for idx in range(6):
            v = np.where(ccdid==idx)
            xpixrot[v]=(xpix[v]-511.5-24)*np.cos(ccdrot[idx])-(ypix[v]-511.5- 9)*np.sin(ccdrot[idx])
            ypixrot[v]=(xpix[v]-511.5-24)*np.sin(ccdrot[idx])+(ypix[v]-511.5- 9)*np.cos(ccdrot[idx])

        xc = ccdcentr - ypixrot*ccdpxsz
        yc =      0.0 + xpixrot*ccdpxsz
        an = -60.0*dsc[:,0]*np.pi/180.0
        xd = xc*np.cos(an)-yc*np.sin(an)
        yd = xc*np.sin(an)+yc*np.cos(an)

        for idx in range(6):
            v = np.where(ccdid==idx)
            xd[v] = xd[v]+ccdoffx[idx]
            yd[v] = yd[v]+ccdoffy[idx]
        yd = yd * pfi_parity
            
        outarray = np.block([[ccdid],
                             [spotid],
                             [xd],
                             [yd],
                             [flux],
                             [smma],
                             [smmi],
                             [flag]]).transpose()
        
        return outarray

    def pfipos2agpixel(self, inarray):
        ccdpxsz  = 0.013
        # ccdcentt = np.array([ 0.0, -60.0,-120.0, 180.0, 120.0, 60.0])*np.pi/180.0
        ccdcentr = 241.314

        ccdoffx = np.array([-0.668, +0.081, +0.180, +0.357, +0.138, -0.077])
        ccdoffy = np.array([+0.405, +0.055, +0.357, -0.270, -0.444, -0.067])
        ccdoffx = ccdoffx - np.array([+0.015, +0.025, +0.014, +0.002, +0.022, +0.004]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdoffy = ccdoffy - np.array([-0.013, +0.012, +0.030, +0.013, -0.015, -0.024]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdrot  = np.array([-0.253368, +0.234505, +0.329449, +0.416894, +0.0589071, +0.234977])*np.pi/180.0

        ### inarray
        # ccdid spotid, xd, yd
        ccdid  = inarray[:,0].astype(int)
        spotid = inarray[:,1].astype(int)
        xd     = inarray[:,2]
        yd     = inarray[:,3]

        yd = yd * pfi_parity
        
        for idx in range(6):
            v = np.where(ccdid==idx)
            xd[v] = xd[v]-ccdoffx[idx]
            yd[v] = yd[v]-ccdoffy[idx]

        an = -60.0*ccdid*np.pi/180.0
        xc = xd*np.cos(an)+yd*np.sin(an)
        yc =-xd*np.sin(an)+yd*np.cos(an)
        xpixrot = (     0.0 +yc)/ccdpxsz
        ypixrot = (ccdcentr -xc)/ccdpxsz

        xpix = np.zeros_like(xpixrot)
        ypix = np.zeros_like(ypixrot)
        for idx in range(6):
            v = np.where(ccdid==idx)
            xpix[v] = xpixrot[v]*np.cos(ccdrot[idx])+ypixrot[v]*np.sin(ccdrot[idx]) + 511.5 + 24
            ypix[v] =-xpixrot[v]*np.sin(ccdrot[idx])+ypixrot[v]*np.cos(ccdrot[idx]) + 511.5 +  9

        outarray = np.block([[ccdid],
                             [spotid],
                             [xpix],
                             [ypix]]).transpose()
        return outarray
        
    def agpixel2fppos(self, dsc):
        ccdpxsz  = 0.013
        # ccdcentt = np.array([ 0.0, -60.0,-120.0, 180.0, 120.0, 60.0])*np.pi/180.0
        ccdcentr = 241.314

        ccdoffx = np.array([+0.405, +0.055, +0.357, -0.270, -0.444, -0.067])
        ccdoffy = np.array([-0.668, +0.081, +0.180, +0.357, +0.138, -0.077])
        ccdoffx = ccdoffx - np.array([-0.013, +0.012, +0.030, +0.013, -0.015, -0.024]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdoffy = ccdoffy - np.array([+0.015, +0.025, +0.014, +0.002, +0.022, +0.004]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdrot  = np.array([-0.253368, +0.234505, +0.329449, +0.416894, +0.0589071, +0.234977])*np.pi/180.0
        
        ccdid  = dsc[:,0].astype(int)
        spotid = dsc[:,1].astype(int)
        flux   = dsc[:,2]
        xpix   = dsc[:,3]
        ypix   = dsc[:,4]
        ag_xy  = dsc[:,5]
        ag_xx  = dsc[:,6]
        ag_yy  = dsc[:,7]
        flag   = dsc[:,12]

        smma   = np.sqrt(0.5*(ag_xx+ag_yy)+np.sqrt((0.5*(ag_xx-ag_yy))**2+ag_xy**2))
        smmi   = np.sqrt(0.5*(ag_xx+ag_yy)-np.sqrt((0.5*(ag_xx-ag_yy))**2+ag_xy**2))

        xpixrot = np.zeros_like(xpix)
        ypixrot = np.zeros_like(ypix)
        for idx in range(6):
            v = np.where(ccdid==idx)
            xpixrot[v]=(xpix[v]-511.5-24)*np.cos(ccdrot[idx])-(ypix[v]-511.5- 9)*np.sin(ccdrot[idx])
            ypixrot[v]=(xpix[v]-511.5-24)*np.sin(ccdrot[idx])+(ypix[v]-511.5- 9)*np.cos(ccdrot[idx])

        xc =      0.0 + xpixrot*ccdpxsz
        yc = ccdcentr - ypixrot*ccdpxsz
        an = +60.0*dsc[:,0]*np.pi/180.0
        xd = xc*np.cos(an)-yc*np.sin(an)
        yd = xc*np.sin(an)+yc*np.cos(an)

        for idx in range(6):
            v = np.where(ccdid==idx)
            xd[v] = xd[v]+ccdoffx[idx]
            yd[v] = yd[v]+ccdoffy[idx]

        outarray = np.block([[ccdid],
                             [spotid],
                             [xd],
                             [yd],
                             [flux],
                             [smma],
                             [smmi],
                             [flag]]).transpose()
        
        return outarray

    def fppos2agpixel(self, inarray):
        ccdpxsz  = 0.013
        # ccdcentt = np.array([ 0.0, -60.0,-120.0, 180.0, 120.0, 60.0])*np.pi/180.0
        ccdcentr = 241.314

        ccdoffx = np.array([+0.405, +0.055, +0.357, -0.270, -0.444, -0.067])
        ccdoffy = np.array([-0.668, +0.081, +0.180, +0.357, +0.138, -0.077])
        ccdoffx = ccdoffx - np.array([-0.013, +0.012, +0.030, +0.013, -0.015, -0.024]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdoffy = ccdoffy - np.array([+0.015, +0.025, +0.014, +0.002, +0.022, +0.004]) ## correction by focus 2.70 data only (with CCD Y offset 8 -> 9 / 2022-09-06)
        ccdrot  = np.array([-0.253368, +0.234505, +0.329449, +0.416894, +0.0589071, +0.234977])*np.pi/180.0

        ### inarray
        # ccdid spotid, xd, yd
        ccdid  = inarray[:,0].astype(int)
        spotid = inarray[:,1].astype(int)
        xd     = inarray[:,2]
        yd     = inarray[:,3]

        for idx in range(6):
            v = np.where(ccdid==idx)
            xd[v] = xd[v]-ccdoffx[idx]
            yd[v] = yd[v]-ccdoffy[idx]

        an = +60.0*ccdid*np.pi/180.0
        xc = xd*np.cos(an)+yd*np.sin(an)
        yc =-xd*np.sin(an)+yd*np.cos(an)
        xpixrot = (     0.0 +xc)/ccdpxsz
        ypixrot = (ccdcentr -yc)/ccdpxsz

        xpix = np.zeros_like(xpixrot)
        ypix = np.zeros_like(ypixrot)
        for idx in range(6):
            v = np.where(ccdid==idx)
            xpix[v] = xpixrot[v]*np.cos(ccdrot[idx])+ypixrot[v]*np.sin(ccdrot[idx]) + 511.5 + 24
            ypix[v] =-xpixrot[v]*np.sin(ccdrot[idx])+ypixrot[v]*np.cos(ccdrot[idx]) + 511.5 +  9

        outarray = np.block([[ccdid],
                             [spotid],
                             [xpix],
                             [ypix]]).transpose()
        return outarray
            

    def agarray2momentdifference(self, array, maxellip, maxsize, minsize):
        ##### array 
        ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix] Flag[0 or 1]
        filtered_agarray, _ = PFS.sourceFilter(self, array, maxellip, maxsize, minsize)
        outarray=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        for ccdid in range(1,7):
            array = filtered_agarray[np.where(filtered_agarray[:,0]==ccdid)]
            array_wosp = array[np.where(array[:,7]==0)]
            array_wisp = array[np.where(array[:,7]==1)]

            moment_wosp = np.median((array_wosp[:,5]**2+array_wosp[:,6]**2)*4)
            moment_wisp = np.median((array_wisp[:,5]**2+array_wisp[:,6]**2)*4)

            outarray[ccdid-1]=moment_wosp-moment_wisp

        return outarray

    def momentdifference2focuserror(self, momentdifference):
        # momentdifference [pixel^2]
        # focuserror [mm]
        focuserror = momentdifference * 0.0086 - 0.026

        return focuserror

###
if __name__ == "__main__":
    print('### basic functions for Subaru telescope, POPT2 and PFS AG')

    ### function test code
    # telfile =  sys.argv[1]
    # telarray= np.loadtxt(telfile, dtype={'names': ('DATE-OBS', 'UT-STR', 'RA2000', 'DEC2000', 'ADC-STR', 'INR-STR', 'M2-POS3', 'WAVELEN'),\
    #                                      'formats':('<U10', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12')})
    # dateobs = str(telarray['DATE-OBS'])
    # utstr   = str(telarray['UT-STR'])
    # ra2000  = str(telarray['RA2000'])
    # dec2000 = str(telarray['DEC2000'])
    # adcstr  = float(telarray['ADC-STR'])
    # inrstr  = float(telarray['INR-STR'])
    # m2pos3  = float(telarray['M2-POS3'])
    # wl      = float(telarray['WAVELEN'])
    # datetime= dateobs+"T"+utstr+"Z"
    # t       = at.Time(datetime)
    # coord   = ac.SkyCoord(ra=ra2000, dec=dec2000, unit=(au.hourangle, au.deg),frame='fk5')
    # tel_ra  = coord.ra.degree
    # tel_de  = coord.dec.degree

    # dscfile = sys.argv[2]
    # dsc     = np.loadtxt(dscfile)
    # gscfile1= sys.argv[3]
    # gsc1    = np.loadtxt(gscfile1)
    # gscfile2= sys.argv[4]
    # gsc2    = np.loadtxt(gscfile2)

    # subaru = Subaru()
    # pfs    = PFS()

    # ra_offset,de_offset,inr_offset, mr, md = \
    #     pfs.RADECInRShift(dsc[:,0],\
    #                       dsc[:,1],\
    #                       dsc[:,2],\
    #                       dsc[:,15],\
    #                       gsc1, gsc2)
    # print("# %12.6f %12.6f %12.6f" %(ra_offset*3600,de_offset*3600,inr_offset*3600))

    # for items in mr:
    #     print(items[0],items[1],items[2],items[3],items[4],items[5],items[6],items[7])

    # # ra_offset,de_offset,inr_offset, mr,md = \
    # #     pfs.RADECShift(dsc[:,0],\
    # #                    dsc[:,1],\
    # #                    dsc[:,2],\
    # #                    dsc[:,15],\
    # #                    gsc1, gsc2)
    # # print("# %12.6f %12.6f %12.6f" %(ra_offset*3600,de_offset*3600,inr_offset*3600))
