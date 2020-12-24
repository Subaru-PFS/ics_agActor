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

        # x = -np.sin(t)* np.sin(s)/0.014
        # y = -np.cos(t)* np.sin(s)/0.014

        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans = np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        #x = np.sin(np.arctan(tanx))/0.014
        #y = np.sin(np.arctan(tany))/0.014
        x = tanx/0.014
        y = tany/0.014

        c2  = +2.62647155e+02 +1.20672582e-02*L +2.81291376e-03*L**2 -1.95004252e-02*L**3 +6.62211933e-03*L**4 +(-6.69205215e-07 -4.63719531e-05*D +3.34738200e-04*D**2)*adc**2
        c7  = +3.35051337e+00 +3.77807822e-03*L -5.80701311e-03*L**2 +7.26283670e-04*L**3 -4.85380067e-04*L**4 +(-6.66148637e-08 +2.90341457e-06*D -2.64881410e-05*D**2)*adc**2
        c14 = +2.39640865e-01 +2.19208164e-03*L +1.16310136e-03*L**2 -8.14944507e-04*L**3 +3.40395789e-04*L**4 +(-8.70555356e-09 -6.08579501e-07*D -1.03617965e-05*D**2)*adc**2
        c23 = +2.75372159e-02 -5.20065582e-05*L -7.66966395e-04*L**2 +3.40109185e-04*L**3 -1.50304073e-04*L**4 +(+3.83321543e-08 -2.75372944e-06*D +2.05525769e-05*D**2)*adc**2
        c34 = +1.72669649e-03 +4.61631727e-04*L -4.53786317e-04*L**2 +2.47728154e-04*L**3 -1.09872108e-04*L**4 +(+2.66165100e-08 -2.31203815e-06*D +1.46891600e-05*D**2)*adc**2
        c47 = -1.38477902e-03 +7.60859906e-04*L -2.85170829e-04*L**2 +2.22644774e-04*L**3 -1.05796649e-04*L**4 +(+3.97182752e-08 -2.97655084e-06*D +2.60533051e-05*D**2)*adc**2
        c6  =                                                                                                  +(+1.17209603e-04 -1.80024562e-02*D +3.58635396e-03*D**2)*adc
        c13 =                                                                                                  +(-2.05135805e-06 -6.10958747e-04*D -2.47540188e-03*D**2)*adc

        c3  = +2.62647155e+02 +1.20672582e-02*L +2.81291378e-03*L**2 -1.95004252e-02*L**3 +6.62211932e-03*L**4 +(-2.59360768e-06 -2.45264542e-04*D +1.46966720e-03*D**2)*adc**2
        c8  = +3.35051337e+00 +3.77807822e-03*L -5.80701311e-03*L**2 +7.26283670e-04*L**3 -4.85380067e-04*L**4 +(-2.79858548e-07 -1.11560705e-05*D +2.47349765e-05*D**2)*adc**2
        c15 = +2.39640865e-01 +2.19208164e-03*L +1.16310136e-03*L**2 -8.14944507e-04*L**3 +3.40395789e-04*L**4 +(-5.87738734e-08 -9.88794442e-07*D -6.23526927e-05*D**2)*adc**2
        c24 = +2.75372159e-02 -5.20065582e-05*L -7.66966395e-04*L**2 +3.40109185e-04*L**3 -1.50304073e-04*L**4 +(+5.01388886e-08 -7.03312952e-06*D +1.96276499e-05*D**2)*adc**2
        c35 = +1.72669649e-03 +4.61631727e-04*L -4.53786317e-04*L**2 +2.47728154e-04*L**3 -1.09872108e-04*L**4 +(+1.21273431e-07 -9.90167250e-06*D +8.05966925e-05*D**2)*adc**2
        c48 = -1.38477902e-03 +7.60859906e-04*L -2.85170829e-04*L**2 +2.22644774e-04*L**3 -1.05796649e-04*L**4 +(+2.56791267e-07 -1.87556960e-05*D +2.10817052e-04*D**2)*adc**2
        c5  =                                                                                                  +(-1.16802867e-04 +1.78828652e-02*D -1.68676967e-03*D**2)*adc
        c12 =                                                                                                  +(+2.93855412e-06 +4.52221307e-04*D +5.18692032e-03*D**2)*adc

        telx = \
            c2*(x) +\
            c7*((3*(x**2+y**2)-2)*x) +\
            c14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            c23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            c34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            c47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            c6*(2*x*y) +\
            c13*((4*(x**2+y**2)-3)*2*x*y)

        tely = \
            c3*(y) +\
            c8*((3*(x**2+y**2)-2)*y) +\
            c15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            c24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            c35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            c48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            c5*(x**2-y**2) +\
            c12*((4*(x**2+y**2)-3)*(x**2-y**2))

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

        c2  = +1.43748364e-02 -6.47205311e-07*L -1.24364376e-07*L**2 +1.00536902e-06*L**3 -3.40360006e-07*L**4 +(+2.20720275e-11 +2.93468854e-09*D -2.65015343e-08*D**2)*adc**2
        c7  = -1.88216196e-04 -1.87107207e-07*L +3.18664532e-07*L**2 -9.76824181e-08*L**3 +4.58863956e-08*L**4 +(+1.28791658e-12 -4.00067839e-10*D +3.58240387e-09*D**2)*adc**2
        c14 = -6.91101676e-06 -1.12018305e-07*L -7.07939788e-08*L**2 +3.59418869e-08*L**3 -1.58019565e-08*L**4 +(-2.14591088e-13 +6.11573661e-11*D +5.41369665e-10*D**2)*adc**2
        c23 = -6.01380804e-07 +8.41284611e-11*L +5.79084526e-08*L**2 -2.92467294e-08*L**3 +1.27994087e-08*L**4 +(-2.04519355e-12 +1.60037945e-10*D -9.59724682e-10*D**2)*adc**2
        c34 = +7.63697864e-08 -3.73597168e-08*L +2.47802638e-08*L**2 -1.48531316e-08*L**3 +6.65531634e-09*L**4 +(-1.09694171e-12 +1.14601042e-10*D -5.14243588e-10*D**2)*adc**2
        c47 = +8.03184434e-08 -4.00579115e-08*L +1.19433347e-08*L**2 -1.03215825e-08*L**3 +5.01515463e-09*L**4 +(-1.69304708e-12 +1.27080679e-10*D -1.02073230e-09*D**2)*adc**2
        c6  =                                                                                                  +(-5.60517199e-09 +9.55298721e-07*D -2.42529747e-07*D**2)*adc
        c13 =                                                                                                  +(+6.90939709e-10 -1.64330245e-08*D +1.34059317e-07*D**2)*adc

        c3  = +1.43748364e-02 -6.47205311e-07*L -1.24364375e-07*L**2 +1.00536902e-06*L**3 -3.40360007e-07*L**4 +(+1.33592956e-10 +1.30880744e-08*D -7.91615707e-08*D**2)*adc**2
        c8  = -1.88216196e-04 -1.87107207e-07*L +3.18664532e-07*L**2 -9.76824181e-08*L**3 +4.58863956e-08*L**4 +(+8.33521141e-12 +3.47944605e-12*D +3.44750441e-09*D**2)*adc**2
        c15 = -6.91101676e-06 -1.12018305e-07*L -7.07939788e-08*L**2 +3.59418869e-08*L**3 -1.58019565e-08*L**4 +(-3.72826287e-13 +2.19897221e-10*D +2.12353646e-09*D**2)*adc**2
        c24 = -6.01380804e-07 +8.41284608e-11*L +5.79084526e-08*L**2 -2.92467294e-08*L**3 +1.27994087e-08*L**4 +(-5.82861489e-12 +6.29402795e-10*D -3.51472083e-09*D**2)*adc**2
        c35 = +7.63697864e-08 -3.73597168e-08*L +2.47802638e-08*L**2 -1.48531316e-08*L**3 +6.65531634e-09*L**4 +(-9.55539007e-12 +7.62414314e-10*D -7.26111058e-09*D**2)*adc**2
        c48 = +8.03184434e-08 -4.00579115e-08*L +1.19433347e-08*L**2 -1.03215825e-08*L**3 +5.01515463e-09*L**4 +(-1.47189264e-11 +1.07314133e-09*D -1.27891321e-08*D**2)*adc**2
        c5  =                                                                                                  +(+5.57543252e-09 -9.48104383e-07*D +1.29131603e-07*D**2)*adc
        c12 =                                                                                                  +(-7.42712508e-10 +2.48523116e-08*D -2.74935643e-07*D**2)*adc

        tantx = \
            c2*(x) +\
            c7*((3*(x**2+y**2)-2)*x) +\
            c14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            c23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            c34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            c47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            c6*(2*x*y) +\
            c13*((4*(x**2+y**2)-3)*2*x*y)

        tanty = \
            c3*(y) +\
            c8*((3*(x**2+y**2)-2)*y) +\
            c15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            c24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            c35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            c48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            c5*(x**2-y**2) +\
            c12*((4*(x**2+y**2)-3)*(x**2-y**2))

        #s = np.arctan(np.sqrt(sintx**2+sinty**2))
        #t = np.arctan2(-sintx, -sinty)
        #tantx = np.tan(np.arcsin(sintx))
        #tanty = np.tan(np.arcsin(sinty))
        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)

        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def sourceFilter(self, agarray):
        ag_smma = agarray[:,5]
        ag_smmi = agarray[:,6]

        # ellipticity condition
        cellip = (1.0-ag_smmi/ag_smma)    < 0.2
        # size condition (upper)
        csizeU = np.sqrt(ag_smmi*ag_smma) < 5.0
        # size condition (lower)
        csizeL = np.sqrt(ag_smmi*ag_smma) > 1.5

        v = cellip*csizeU*csizeL

        return v

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

        return ra_offset, de_offset, inr_offset, f, min_dist_index[f], errx, erry

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
