# -*- coding: utf-8 -*-
import numpy as np

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance, EarthLocation, AltAz, Angle
from astropy.utils import iers
iers.conf.auto_download = True

d_ra  = 1.0/3600.0
d_de  = 1.0/3600.0
d_inr = 0.01

### constants proper to HSC camera
# hsc_inr_zero_offset        = +0.07 # in degree
# hsc_detector_zero_offset_x = -0.15 # in mm
# hsc_detector_zero_offset_y = +0.18 # in mm
hsc_inr_zero_offset        = 0 # in degree
hsc_detector_zero_offset_x = 0 # in mm
hsc_detector_zero_offset_y = 0 # in mm

class Subaru():
    def starSepZPA(self, tel_ra, tel_de, str_ra, str_de, t):
        tel_ra = tel_ra*u.degree
        tel_de = tel_de*u.degree
        str_ra = str_ra*u.degree
        str_de = str_de*u.degree

        tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='icrs')
        str_coord = SkyCoord(ra=str_ra, dec=str_de, frame='icrs')

        # sbr = EarthLocation.of_site('Subaru Telescope')
        sbr = EarthLocation(lat=Angle((19, 49, 31.8), unit=u.deg), lon=Angle((-155, 28, 33.7), unit=u.deg), height=4163)
        frame_subaru = AltAz(obstime  = t, location = sbr,\
                             pressure = 620*u.hPa, obswl = 0.77*u.micron)

        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = str_coord.transform_to(frame_subaru)

        str_sep =  tel_altaz.separation(str_altaz).degree
        str_zpa = -tel_altaz.position_angle(str_altaz).degree

        return str_sep, str_zpa

    def starSepZPAGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t):
        tel_ra = tel_ra*u.degree
        tel_de = tel_de*u.degree
        str_ra = str_ra*u.degree
        str_de = str_de*u.degree

        tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='icrs', obstime=t)
        str_coord = SkyCoord(ra=str_ra, dec=str_de,
                             distance=Distance(parallax=str_plx * u.mas, allow_negative=True),
                             pm_ra_cosdec=str_pmRA * u.mas/u.yr,
                             pm_dec=str_pmDE * u.mas/u.yr,
                             obstime=Time(2015.5, format='decimalyear'),
                             frame='icrs')

        # sbr = EarthLocation.of_site('Subaru Telescope')
        sbr = EarthLocation(lat=Angle((19, 49, 31.8), unit=u.deg), lon=Angle((-155, 28, 33.7), unit=u.deg), height=4163)
        frame_subaru = AltAz(obstime  = t, location = sbr,\
                             pressure = 620*u.hPa, obswl = 0.77*u.micron)

        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = str_coord.transform_to(frame_subaru)

        str_sep =  tel_altaz.separation(str_altaz).degree
        str_zpa = -tel_altaz.position_angle(str_altaz).degree

        return str_sep, str_zpa

    def radec2radecplxpm(self, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t):
        str_plx[np.where(str_plx<0.00001)]=0.00001

        str_coord = SkyCoord(ra=str_ra*u.degree, dec=str_de*u.degree,
                             distance=Distance(parallax=str_plx * u.mas, allow_negative=False),
                             pm_ra_cosdec=str_pmRA * u.mas/u.yr,
                             pm_dec= str_pmDE * u.mas/u.yr,
                             obstime=Time(2015.5, format='decimalyear'),
                             frame='icrs')

        str_coord_obstime = str_coord.apply_space_motion(Time(t))

        ra = str_coord_obstime.ra.degree
        de = str_coord_obstime.dec.degree

        return ra,de

class POPT2():
    def telStatusfromText(self, textfile):
        array = np.loadtxt(textfile,\
                           dtype={\
                               'names': ('DATE-OBS', 'UT-STR', 'RA2000', 'DEC2000', 'ADC-STR', 'INR-STR'),\
                               'formats':('<U10', '<U12', '<U12', '<U12', '<U12', '<U12')})
        dateobs = str(array['DATE-OBS'])
        utstr   = str(array['UT-STR'])
        ra2000  = str(array['RA2000'])
        dec2000 = str(array['DEC2000'])
        adcstr  = float(array['ADC-STR'])
        inrstr  = float(array['INR-STR'])

        datetime= dateobs+"T"+utstr+"Z"
        coord = SkyCoord(ra=ra2000, dec=dec2000, unit=(u.hourangle, u.deg),
                         frame='icrs')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree

        return tel_ra, tel_de, datetime, adcstr, inrstr

    def celestial2focalplane(self, sep, zpa, adc):
        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is sin(s) < 0.014 (equiv. to 0.80216712 degree)
        x = -np.sin(t)* np.sin(s)/0.014
        y = -np.cos(t)* np.sin(s)/0.014

        # HSC-i
        # c2 = 2.62712058958421e+02 -1.75552949394474e-06*adc**2
        # c7 = 3.36363419130444e+00 -7.77258745608301e-08*adc**2
        # c14= 2.39556420581842e-01 -1.34765101243770e-07*adc**2
        # c23= 2.74199565969097e-02 -1.06998796205918e-07*adc**2
        # c10= 5.24182514647881e-05 +8.88141736131490e-08*adc**2
        # c6 = 0.000671476724227082*adc

        # c3 = 2.62712056260913e+02 -4.54965104302589e-05*adc**2
        # c8 = 3.36362932755874e+00 -6.52163563633518e-05*adc**2
        # c15= 2.39549423378678e-01 -7.53113039413331e-05*adc**2
        # c24= 2.74116747515975e-02 -6.65916863956645e-05*adc**2
        # c11=-5.27438382354667e-05 -3.28967242481091e-06*adc**2
        # c5 =-0.000663853081621188*adc

        # HSC-r
        c2 =  2.62713997e+02 -1.74358206e-06*adc**2
        c7 =  3.36646746e+00 -2.88958580e-09*adc**2
        c14=  2.41063109e-01 -9.82519927e-08*adc**2
        c23=  2.86910313e-02 -1.81538844e-07*adc**2
        c10=  2.39613061e-04 +1.21711231e-07*adc**2
        c6 =  7.52927675e-04*adc

        c3 =  2.62713997e+02 -9.83988677e-06*adc**2
        c8 =  3.36646746e+00 -6.18741722e-06*adc**2
        c15=  2.41063109e-01 -1.68825349e-05*adc**2
        c24=  2.86910313e-02 -4.14132732e-05*adc**2
        c11= -2.39613061e-04 -6.83967290e-07*adc**2
        c5 = -7.47538979e-04*adc

        telx = \
            c2*(x) +\
            c7*((3*(x**2+y**2)-2)*x) +\
            c14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            c23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            c10*((x**2-3*y**2)*x) +\
            c6*(2*x*y)
        tely = \
            c3*(y) +\
            c8*((3*(x**2+y**2)-2)*y) +\
            c15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            c24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            c11*((3*x**2-y**2)*y) +\
            c5*(x**2-y**2)

        return telx,tely

    def focalplane2celestial(self, xt, yt, adc):
        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        # HSC-i
        # c2  = 1.43721073e-2 +8.45742896e-11*adc**2
        # c7  =-1.88481512e-4 -8.54773812e-12*adc**2
        # c14 =-6.99524470e-6 +6.75921244e-13*adc**2
        # c23 =-7.26098879e-7 +2.53719080e-13*adc**2
        # c10 = 3.69046017e-7 +4.40131478e-12*adc**2
        # c6  = 2.21641168e-8*adc

        # c3  = 1.43721073e-2 +4.46308029e-10*adc**2
        # c8  =-1.88481512e-4 +6.90663074e-12*adc**2
        # c15 =-6.99524470e-6 +2.25955460e-12*adc**2
        # c24 =-7.26098879e-7 +3.43957790e-12*adc**2
        # c11 =-3.69046017e-7 +4.55537365e-12*adc**2
        # c5  =-2.21539988e-8*adc

        # HSC-r
        c2  = 1.43720661e-02 +8.73098647e-11*adc**2
        c7  =-1.88505586e-04 -7.29895287e-12*adc**2
        c14 =-6.90286146e-06 +2.95845304e-12*adc**2
        c23 =-6.10509443e-07 +2.98407611e-12*adc**2
        c10 = 3.52068898e-07 +3.82263321e-12*adc**2
        c6  = 2.36143951e-08*adc

        c3  = 1.43720660e-02 +4.56838387e-10*adc**2
        c8  =-1.88505600e-04 +2.25907820e-11*adc**2
        c15 =-6.90284586e-06 +2.92610941e-11*adc**2
        c24 =-6.10447604e-07 +4.30181634e-11*adc**2
        c11 =-3.52062707e-07 +2.72749282e-12*adc**2
        c5  =-2.33301629e-08*adc

        tantx = \
            c2*(x) +\
            c7*((3*(x**2+y**2)-2)*x) +\
            c14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            c23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            c10*((x**2-3*y**2)*x) +\
            c6*(2*x*y)
        tanty = \
            c3*(y) +\
            c8*((3*(x**2+y**2)-2)*y) +\
            c15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            c24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            c11*((3*x**2-y**2)*y) +\
            c5*(x**2-y**2)

        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)

        return s, t

    def sourceFilter(self, agarray):
        ag_ccd  = agarray[:,0]
        ag_id   = agarray[:,1]
        ag_xc   = agarray[:,2]
        ag_yc   = agarray[:,3]
        ag_flx  = agarray[:,4]
        ag_smma = agarray[:,5]
        ag_smmi = agarray[:,6]

        # ellipticity condition
        cellip = (1.0-ag_smmi/ag_smma)    < 0.2
        # size condition (upper)
        csizeU = np.sqrt(ag_smmi*ag_smma) < 5.0
        # size condition (lower)
        csizeL = np.sqrt(ag_smmi*ag_smma) > 1.5

        v = cellip*csizeU*csizeL

        vdatanum = np.sum(v)

        oarray = np.zeros((vdatanum,7))
        oarray[:,0] = ag_ccd[v]
        oarray[:,1] = ag_id[v]
        oarray[:,2] = ag_xc[v]
        oarray[:,3] = ag_yc[v]
        oarray[:,4] = ag_flx[v]
        oarray[:,5] = ag_smma[v]
        oarray[:,6] = ag_smmi[v]

        return oarray

    def RADECInRShift(self, obj_xdp, obj_ydp, obj_int, \
                      cat_xdp, cat_ydp, cat_mag,\
                      dxra,dyra,dxde,dyde,dxinr,dyinr):
        n_obj = (obj_xdp.shape)[0]
        xdiff = np.transpose([obj_xdp])-cat_xdp
        ydiff = np.transpose([obj_ydp])-cat_ydp
        dist  = np.sqrt(xdiff**2+ydiff**2)
        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index
        rCRA = np.median((xdiff[min_dist_indices]*dyde[min_dist_index]-ydiff[min_dist_indices]*dxde[min_dist_index])/(dxra[min_dist_index]*dyde[min_dist_index]-dyra[min_dist_index]*dxde[min_dist_index]))
        rCDE = np.median((xdiff[min_dist_indices]*dyra[min_dist_index]-ydiff[min_dist_indices]*dxra[min_dist_index])/(dxde[min_dist_index]*dyra[min_dist_index]-dyde[min_dist_index]*dxra[min_dist_index]))

        xdiff = np.transpose([obj_xdp])-(cat_xdp+rCRA*dxra+rCDE*dxde)
        ydiff = np.transpose([obj_ydp])-(cat_ydp+rCRA*dyra+rCDE*dyde)

        dist  = np.sqrt(xdiff**2+ydiff**2)

        min_dist_index   = np.nanargmin(dist, axis=1)
        min_dist_indices = np.array(range(n_obj)),min_dist_index

        f = dist[min_dist_indices]<0.2

        match_obj_xdp = obj_xdp[f]
        match_obj_ydp = obj_ydp[f]
        match_obj_int = obj_int[f]
        match_cat_xdp = (cat_xdp[min_dist_index])[f]
        match_cat_ydp = (cat_ydp[min_dist_index])[f]
        match_cat_mag = (cat_mag[min_dist_index])[f]
        match_dxra  = (dxra[min_dist_index])[f]
        match_dyra  = (dyra[min_dist_index])[f]
        match_dxde  = (dxde[min_dist_index])[f]
        match_dyde  = (dyde[min_dist_index])[f]
        match_dxinr = (dxinr[min_dist_index])[f]
        match_dyinr = (dyinr[min_dist_index])[f]

        dra  = np.concatenate([match_dxra,match_dyra])
        dde  = np.concatenate([match_dxde,match_dyde])
        dinr = np.concatenate([match_dxinr,match_dyinr])

        basis= np.stack([dra,dde,dinr]).transpose()

        errx = match_obj_xdp - match_cat_xdp
        erry = match_obj_ydp - match_cat_ydp
        err  = np.concatenate([errx,erry])

        M = np.zeros((3,3))
        b = np.zeros((3))
        for itr1 in range(3):
            for itr2 in range(3):
                M[itr1,itr2] = np.sum(basis[:,itr1]*basis[:,itr2])
            b[itr1] = np.sum(basis[:,itr1]*err)

        A = np.linalg.solve(M,b)

        ra_offset  = A[0] * d_ra
        de_offset  = A[1] * d_de
        inr_offset = A[2] * d_inr

        return ra_offset, de_offset, inr_offset

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr):
        sep0,zpa0 = Subaru.starSepZPA(self, tel_ra, tel_de, str_ra, str_de, t)
        sep1,zpa1 = Subaru.starSepZPA(self, tel_ra+d_ra, tel_de, str_ra, str_de, t)
        sep2,zpa2 = Subaru.starSepZPA(self, tel_ra, tel_de+d_de, str_ra, str_de, t)

        xfp0,yfp0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc)
        xfp1,yfp1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc)
        xfp2,yfp2 = POPT2.celestial2focalplane(self, sep2,zpa2,adc)

        xdp0,ydp0 = HSC.fp2dp(self, xfp0,yfp0,inr)
        xdp1,ydp1 = HSC.fp2dp(self, xfp1,yfp1,inr)
        xdp2,ydp2 = HSC.fp2dp(self, xfp2,yfp2,inr)
        xdp3,ydp3 = HSC.fp2dp(self, xfp0,yfp0,inr+d_inr)

        dxdpdra = xdp1-xdp0
        dydpdra = ydp1-ydp0
        dxdpdde = xdp2-xdp0
        dydpdde = ydp2-ydp0
        dxdpdinr= xdp3-xdp0
        dydpdinr= ydp3-ydp0

        return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr

    def makeBasisGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t, adc, inr):
        sep0,zpa0 = Subaru.starSepZPAGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)
        sep1,zpa1 = Subaru.starSepZPAGaia(self, tel_ra+d_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)
        sep2,zpa2 = Subaru.starSepZPAGaia(self, tel_ra, tel_de+d_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)

        xfp0,yfp0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc)
        xfp1,yfp1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc)
        xfp2,yfp2 = POPT2.celestial2focalplane(self, sep2,zpa2,adc)

        xdp0,ydp0 = HSC.fp2dp(self, xfp0,yfp0,inr)
        xdp1,ydp1 = HSC.fp2dp(self, xfp1,yfp1,inr)
        xdp2,ydp2 = HSC.fp2dp(self, xfp2,yfp2,inr)
        xdp3,ydp3 = HSC.fp2dp(self, xfp0,yfp0,inr+d_inr)

        dxdpdra = xdp1-xdp0
        dydpdra = ydp1-ydp0
        dxdpdde = xdp2-xdp0
        dydpdde = ydp2-ydp0
        dxdpdinr= xdp3-xdp0
        dydpdinr= ydp3-ydp0

        return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr

class HSC():
    def fp2dp(self, xt, yt, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xt*np.cos(inr)+yt*np.sin(inr))+hsc_detector_zero_offset_x
        y = (xt*np.sin(inr)-yt*np.cos(inr))+hsc_detector_zero_offset_y

        return x,y

    def dp2fp(self, xc, yc, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xc-hsc_detector_zero_offset_x)*np.cos(inr)+(yc-hsc_detector_zero_offset_y)*np.sin(inr)
        y = (xc-hsc_detector_zero_offset_x)*np.sin(inr)-(yc-hsc_detector_zero_offset_y)*np.cos(inr)

        return x,y

###
if __name__ == "__main__":
    print('basic functions for Subaru telescope, POPT2 and HSC')
