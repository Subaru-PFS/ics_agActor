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

### constants proper to WFC optics
wfc_scale_temp_coeff = -1.42e-05

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
                               'names': ('DATE-OBS', 'UT-STR', 'RA2000', 'DEC2000', 'ADC-STR', 'INR-STR', 'DOM-TMP', 'WAVELEN'),\
                               'formats':('<U10', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12')})
        dateobs = str(array['DATE-OBS'])
        utstr   = str(array['UT-STR'])
        ra2000  = str(array['RA2000'])
        dec2000 = str(array['DEC2000'])
        adcstr  = float(array['ADC-STR'])
        inrstr  = float(array['INR-STR'])
        domtmp  = float(array['DOM-TMP'])
        wl      = float(array['WAVELEN'])

        datetime= dateobs+"T"+utstr+"Z"
        coord = SkyCoord(ra=ra2000, dec=dec2000, unit=(u.hourangle, u.deg),
                         frame='icrs')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree

        return tel_ra, tel_de, datetime, adcstr, inrstr, domtmp, wl

    def celestial2focalplane(self, sep, zpa, adc, envtmp=273.15, wl=0.77):
        ## index difference PBL1Y_OLD - BSL7Y
        dn=0.0269+0.000983/(wl-0.118)**2
        D=dn

        ## index of silica
        K1=6.961663e-01
        L1=4.679148e-03
        K2=4.079426e-01
        L2=1.351206e-02
        K3=8.974794e-01
        L3=9.793400e+01
        ns=np.sqrt(1+K1*wl**2/(wl**2-L1)+K2*wl**2/(wl**2-L2)+K3*wl**2/(wl**2-L3))

        ## band parameter
        L=(ns-1.458)*100.0

        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is sin(s) < 0.014 (equiv. to 0.80216712 degree)
        x = -np.sin(t)* np.sin(s)/0.014
        y = -np.cos(t)* np.sin(s)/0.014

        c2  = +2.62667525e+02 +6.79629298e-03*L +2.95485180e-03*L**2 -1.24427501e-02*L**3 +4.29538574e-03*L**4     +(-1.02896197e-06 -2.44232480e-05*D)*adc**2
        c7  = +3.36066057e+00 +2.91618032e-03*L -5.94570080e-03*L**2 +2.25906864e-03*L**3 -9.54318056e-04*L**4     +(-3.14122633e-08 +5.93271969e-07*D)*adc**2
        c14 = +2.40282056e-01 +2.08597289e-03*L +8.73164378e-04*L**2 +5.17043407e-05*L**3 +1.05290296e-04*L**4     +(+4.63022273e-09 -2.03620979e-06*D)*adc**2
        c23 = +2.79669736e-02 -3.82625311e-04*L -1.21946258e-03*L**2 +1.99189756e-03*L**3 -6.43291673e-04*L**4     +(+1.67178727e-08 -2.51789315e-06*D)*adc**2
        c10 = +1.78766516e-04 +2.61041832e-05*L -3.85742701e-05*L**2 +1.11526349e-04*L**3 -3.40352999e-05*L**4     +(+7.00048528e-08 +5.29619686e-07*D)*adc**2
        c6  =                                                                                                      +(+1.13257505e-04 -1.79098816e-02*D)*adc

        c3  = +2.62667525e+02 +6.79629298e-03*L +2.95485181e-03*L**2 -1.24427501e-02*L**3 +4.29538573e-03*L**4     +(-4.19817951e-06 -1.48072020e-04*D)*adc**2
        c8  = +3.36066057e+00 +2.91618032e-03*L -5.94570080e-03*L**2 +2.25906864e-03*L**3 -9.54318056e-04*L**4     +(-2.90664180e-07 -1.14168012e-05*D)*adc**2
        c15 = +2.40282056e-01 +2.08597289e-03*L +8.73164378e-04*L**2 +5.17043406e-05*L**3 +1.05290296e-04*L**4     +(+3.03392976e-08 -8.36448773e-06*D)*adc**2
        c24 = +2.79669736e-02 -3.82625311e-04*L -1.21946258e-03*L**2 +1.99189756e-03*L**3 -6.43291673e-04*L**4     +(+5.64175587e-08 -1.08834836e-05*D)*adc**2
        c11 = -1.78766516e-04 -2.61041832e-05*L +3.85742701e-05*L**2 -1.11526349e-04*L**3 +3.40352999e-05*L**4     +(+3.09339180e-08 +2.51153604e-06*D)*adc**2
        c5  =                                                                                                      +(-1.15297622e-04 +1.79322188e-02*D)*adc

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

        telx = telx * (1.0+(envtmp-273.15)*wfc_scale_temp_coeff)
        tely = tely * (1.0+(envtmp-273.15)*wfc_scale_temp_coeff)

        return telx,tely

    def focalplane2celestial(self, xt, yt, adc, envtmp=273.15, wl=0.77):
        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(envtmp-273.15)*wfc_scale_temp_coeff)
        y = y / (1.0+(envtmp-273.15)*wfc_scale_temp_coeff)

        ## index difference PBL1Y_OLD - BSL7Y
        dn=0.0269+0.000983/(wl-0.118)**2
        D=dn

        ## index of silica
        K1=6.961663e-01
        L1=4.679148e-03
        K2=4.079426e-01
        L2=1.351206e-02
        K3=8.974794e-01
        L3=9.793400e+01
        ns=np.sqrt(1+K1*wl**2/(wl**2-L1)+K2*wl**2/(wl**2-L2)+K3*wl**2/(wl**2-L3))

        ## band parameter
        L=(ns-1.458)*100.0

        c2  = +1.43744915e-02 -3.66089499e-07*L -1.32173994e-07*L**2 +6.30310359e-07*L**3 -2.17492601e-07*L**4  +(+5.12241706e-11 +1.16857741e-09*D)*adc**2
        c7  = -1.88330903e-04 -1.46551865e-07*L +3.24759111e-07*L**2 -1.68546415e-07*L**3 +6.63343214e-08*L**4  +(-2.91612501e-12 -1.40785167e-10*D)*adc**2
        c14 = -6.88533077e-06 -9.59400370e-08*L -6.00088739e-08*L**2 -1.38481412e-08*L**3 -2.99932745e-09*L**4  +(-7.72671402e-13 +1.18001709e-10*D)*adc**2
        c23 = -5.76217456e-07 +3.12412231e-08*L +7.04529374e-08*L**2 -1.08409015e-07*L**3 +3.56485140e-08*L**4  +(-9.09493515e-13 +1.21275883e-10*D)*adc**2
        c10 =  3.55386763e-07 -1.38438066e-09*L +1.97735820e-09*L**2 -5.70585703e-09*L**3 +1.73697244e-09*L**4  +(+4.74196770e-14 +1.18597848e-10*D)*adc**2
        c6  =                                                                                                   +(-5.28327182e-09 +9.38352100e-07*D)*adc

        c3  = +1.43744915e-02 -3.66089499e-07*L -1.32173994e-07*L**2 +6.30310359e-07*L**3 -2.17492601e-07*L**4  +(+2.20632222e-10 +7.82650662e-09*D)*adc**2
        c8  = -1.88330903e-04 -1.46551865e-07*L +3.24759111e-07*L**2 -1.68546415e-07*L**3 +6.63343214e-08*L**4  +(+3.80828224e-12 +3.14914474e-10*D)*adc**2
        c15 = -6.88533077e-06 -9.59400370e-08*L -6.00088739e-08*L**2 -1.38481412e-08*L**3 -2.99932745e-09*L**4  +(-3.40551222e-12 +4.81036157e-10*D)*adc**2
        c24 = -5.76217456e-07 +3.12412231e-08*L +7.04529374e-08*L**2 -1.08409015e-07*L**3 +3.56485140e-08*L**4  +(-2.68215298e-12 +5.45580678e-10*D)*adc**2
        c11 = -3.55386763e-07 +1.38438066e-09*L -1.97735820e-09*L**2 +5.70585703e-09*L**3 -1.73697244e-09*L**4  +(+2.07735825e-12 +1.46216935e-11*D)*adc**2
        c5  =                                                                                                   +(+5.37984880e-09 -9.38735856e-07*D)*adc

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

        s = np.rad2deg(s)
        t = np.rad2deg(t)

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

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, envtmp, wl):
        sep0,zpa0 = Subaru.starSepZPA(self, tel_ra, tel_de, str_ra, str_de, t)
        sep1,zpa1 = Subaru.starSepZPA(self, tel_ra+d_ra, tel_de, str_ra, str_de, t)
        sep2,zpa2 = Subaru.starSepZPA(self, tel_ra, tel_de+d_de, str_ra, str_de, t)

        xfp0,yfp0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc,envtmp,wl)
        xfp1,yfp1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc,envtmp,wl)
        xfp2,yfp2 = POPT2.celestial2focalplane(self, sep2,zpa2,adc,envtmp,wl)

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
