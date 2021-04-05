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

### unknown scale factor
# Unknown_Scale_Factor = 1.0 + 2.22e-4
Unknown_Scale_Factor = 1.0

### constants proper to WFC optics
# wfc_scale_temp_coeff = -1.42e-05
wfc_scale_M2POS3_coeff = 1.01546e-4

### constants proper to PFS camera
pfs_inr_zero_offset        =  0.00 # in degree
pfs_detector_zero_offset_x =  0.00 # in mm
pfs_detector_zero_offset_y =  0.00 # in mm

class Misc():
    def diff_index_pbl1yold_bsl7y(self, wl):
        dn=0.0269+0.000983/(wl-0.118)**2 # used only unit conversion... precision is not very high.
        return dn

class Subaru():
    def starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t):
        tel_ra = tel_ra*u.degree
        tel_de = tel_de*u.degree
        str_ra = str_ra*u.degree
        str_de = str_de*u.degree

        tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='icrs')
        str_coord = SkyCoord(ra=str_ra, dec=str_de, frame='icrs')

        #sbr = EarthLocation.of_site('Subaru Telescope')
        sbr = EarthLocation(lat=Angle((19, 49, 31.8), unit=u.deg), lon=Angle((-155, 28, 33.7), unit=u.deg), height=4163)
        frame_subaru = AltAz(obstime  = t, location = sbr,\
                             pressure = 620*u.hPa, obswl = wl*u.micron)

        tel_altaz = tel_coord.transform_to(frame_subaru)
        str_altaz = str_coord.transform_to(frame_subaru)

        str_sep =  tel_altaz.separation(str_altaz).degree
        str_zpa = -tel_altaz.position_angle(str_altaz).degree

        return str_sep, str_zpa

    # def starSepZPAGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, wl, t):
    #     tel_ra = tel_ra*u.degree
    #     tel_de = tel_de*u.degree
    #     str_ra = str_ra*u.degree
    #     str_de = str_de*u.degree

    #     tel_coord = SkyCoord(ra=tel_ra, dec=tel_de, frame='icrs', obstime=t)
    #     str_coord = SkyCoord(ra=str_ra, dec=str_de,
    #                          distance=Distance(parallax=str_plx * u.mas, allow_negative=True),
    #                          pm_ra_cosdec=str_pmRA * u.mas/u.yr,
    #                          pm_dec=str_pmDE * u.mas/u.yr,
    #                          obstime=Time(2015.5, format='decimalyear'),
    #                          frame='icrs')

    #     sbr = EarthLocation.of_site('Subaru Telescope')
    #     frame_subaru = AltAz(obstime  = t, location = sbr,\
    #                          pressure = 620*u.hPa, obswl = wl*u.micron)

    #     tel_altaz = tel_coord.transform_to(frame_subaru)
    #     str_altaz = str_coord.transform_to(frame_subaru)

    #     str_sep =  tel_altaz.separation(str_altaz).degree
    #     str_zpa = -tel_altaz.position_angle(str_altaz).degree

    #     return str_sep, str_zpa

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
                               'names': ('DATE-OBS', 'UT-STR', 'RA2000', 'DEC2000', 'ADC-STR', 'INR-STR', 'M2-POS3', 'WAVELEN'),\
                               'formats':('<U10', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12', '<U12')})
        dateobs = str(array['DATE-OBS'])
        utstr   = str(array['UT-STR'])
        ra2000  = str(array['RA2000'])
        dec2000 = str(array['DEC2000'])
        adcstr  = float(array['ADC-STR'])
        inrstr  = float(array['INR-STR'])
        m2pos3  = float(array['M2-POS3'])
        wl      = float(array['WAVELEN'])

        datetime= dateobs+"T"+utstr+"Z"
        coord = SkyCoord(ra=ra2000, dec=dec2000, unit=(u.hourangle, u.deg),
                         frame='icrs')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree

        return tel_ra, tel_de, datetime, adcstr, inrstr, m2pos3, wl

    def celestial2focalplane(self, sep, zpa, adc, m2pos3, wl, flag):
        f = flag.astype(int)

        xfp_wisp, yfp_wisp = POPT2.celestial2focalplane_wisp(self, sep, zpa, adc, m2pos3, wl)
        xfp_wosp, yfp_wosp = POPT2.celestial2focalplane_wosp(self, sep, zpa, adc, m2pos3, wl)

        xfp = xfp_wosp*f + xfp_wisp*(1-f)
        yfp = yfp_wosp*f + yfp_wisp*(1-f)

        return xfp,yfp

    def focalplane2celestial(self, xt, yt, adc, m2pos3, wl, flag):
        f = flag.astype(int)

        r_wisp, t_wisp = POPT2.focalplane2celestial_wisp(self, xt, yt, adc, m2pos3, wl)
        r_wosp, t_wosp = POPT2.focalplane2celestial_wosp(self, xt, yt, adc, m2pos3, wl)

        r = r_wosp*f + r_wisp*(1-f)
        t = t_wosp*f + t_wisp*(1-f)

        return r,t

    def additionaldistortion(self, xt, yt):
        x = xt / 270.0
        y = yt / 270.0

        # data from HSC-PFS test observation on 2020/10/23, 7 sets of InR rotating data.
        adx =  5.96462898e-05 +2.42107822e-03*(2*(x**2+y**2)-1) +3.81085098e-03*(x**2-y**2) -2.75632544e-03*(2*x*y) -1.35748905e-03*((3*(x**2+y**2)-2)*x) +2.12301241e-05*((3*(x**2+y**2)-2)*y) -9.23710861e-05*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)
        ady = -2.06476363e-04 -3.45168908e-03*(2*(x**2+y**2)-1) +4.09091198e-03*(x**2-y**2) +3.41002899e-03*(2*x*y) +1.63045902e-04*((3*(x**2+y**2)-2)*x) -1.01811037e-03*((3*(x**2+y**2)-2)*y) +2.07395905e-04*(6*(x**2+y**2)**2-6*(x**2+y**2)+1)

        return adx,ady

    def celestial2focalplane_wisp(self, sep, zpa, adc, m2pos3, wl):
        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx2 =  2.62676925e+02 +(-1.87221607e-06 +2.92222089e-06*D)*adc**2
        cx7 =  3.35307347e+00 +(-7.81712660e-09 -1.05819104e-09*D)*adc**2
        cx14=  2.38166168e-01 +(-2.03764107e-08 -7.38480616e-07*D)*adc**2
        cx23=  2.36440726e-02 +( 5.96839761e-09 -1.07538100e-06*D)*adc**2
        cx34= -2.62961444e-03 +( 8.86909851e-09 -9.14461129e-07*D)*adc**2
        cx47=  2.20267977e-03 +(-4.30901437e-08 +1.07839312e-06*D)*adc**2
        cx6 =                  ( 5.03828902e-04 -3.02018655e-02*D)*adc
        cx13=                  ( 1.62039347e-04 -5.25199388e-03*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  2.62676925e+02 +(-8.86455750e-06 +3.15626958e-06*D)*adc**2
        cy8 =  3.35307347e+00 +(-6.26053110e-07 -2.95393149e-07*D)*adc**2
        cy15=  2.38166168e-01 +(-1.50434841e-07 -1.29919261e-06*D)*adc**2
        cy24=  2.36440726e-02 +(-1.29277787e-07 -1.83675834e-06*D)*adc**2
        cy35= -2.62961444e-03 +(-5.72922063e-08 -1.27892174e-06*D)*adc**2
        cy48=  2.20267977e-03 +(-1.11704310e-08 +1.53690637e-06*D)*adc**2
        cy5 =                 +(-4.99201211e-04 +3.00941960e-02*D)*adc
        cy12=                 +(-1.14622807e-04 +4.07269286e-03*D)*adc
        cy1 =                 +( 2.26420481e-02 -7.33125190e-01*D)*adc
        cy4 =                 +( 9.65017704e-04 -3.01504932e-02*D)*adc
        cy9 =                 +( 1.76747572e-04 -4.84455318e-03*D)*adc

        telx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tely = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        telx = telx * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor
        tely = tely * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely

        return telx,tely

    def focalplane2celestial_wisp(self, xt, yt, adc, m2pos3, wl):
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely

        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor

        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        cx2 =  1.43733003e-02 +( 8.72090395e-11 +0.00000000e+00*D)*adc**2
        cx7 = -1.88237022e-04 +(-7.82457037e-12 +1.21758496e-11*D)*adc**2
        cx14= -6.78700730e-06 +( 1.66205167e-12 +1.81935028e-11*D)*adc**2
        cx23= -3.85793541e-07 +( 8.07898607e-13 +2.19246566e-11*D)*adc**2
        cx34=  2.67691166e-07 +( 1.73923452e-13 +1.67246499e-11*D)*adc**2
        cx47= -1.20869185e-07 +( 9.09157007e-13 -2.89301337e-11*D)*adc**2
        cx6 =                 +( 2.29093928e-08 +8.45286748e-09*D)*adc
        cx13=                 +(-2.12409671e-09 +3.88359241e-08*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  1.43733003e-02 +( 4.61943503e-10 +0.00000000e+00*D)*adc**2
        cy8 = -1.88237022e-04 +( 1.26225040e-11 +8.53001285e-12*D)*adc**2
        cy15= -6.78700730e-06 +( 8.50721539e-12 +2.59588237e-11*D)*adc**2
        cy24= -3.85793541e-07 +( 7.98095083e-12 +6.92047432e-11*D)*adc**2
        cy35=  2.67691166e-07 +( 1.36129621e-12 +8.45438550e-11*D)*adc**2
        cy48= -1.20869185e-07 +( 4.48176515e-13 -7.66773681e-11*D)*adc**2
        cy5 =                 +(-2.31614663e-08 -3.71361360e-09*D)*adc
        cy12=                 +( 3.43468003e-10 +2.70608246e-10*D)*adc
        cy1 =                 +(-1.19076885e-06 +3.85419528e-05*D)*adc
        cy4 =                 +(-2.61964016e-09 +5.85399688e-09*D)*adc
        cy9 =                 +(-2.25081043e-09 +2.51123479e-08*D)*adc

        tantx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tanty = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        s = np.arctan(np.sqrt(tantx**2+tanty**2))
        t = np.arctan2(-tantx, -tanty)

        s = np.rad2deg(s)
        t = np.rad2deg(t)

        return s, t

    def celestial2focalplane_wosp(self, sep, zpa, adc, m2pos3, wl):
        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        s = np.deg2rad(sep)
        t = np.deg2rad(zpa)
        # domain is tan(s) < 0.014 (equiv. to 0.8020885128 degree)
        tans =  np.tan(s)
        tanx = -np.sin(t)*tans
        tany = -np.cos(t)*tans
        x = tanx/0.014
        y = tany/0.014

        cx2 =  2.62647800e+02 +(-1.87011330e-06 +2.77589523e-06*D)*adc**2
        cx7 =  3.35086900e+00 +(-7.57865771e-09 +5.53973051e-08*D)*adc**2
        cx14=  2.38964699e-01 +(-1.94709286e-08 -7.73207447e-07*D)*adc**2
        cx23=  2.60230125e-02 +( 6.60545302e-09 -1.16130987e-06*D)*adc**2
        cx34= -1.83104099e-03 +( 1.57806929e-08 -1.02698829e-06*D)*adc**2
        cx47=  2.04091566e-03 +(-4.93364523e-08 +1.22931907e-06*D)*adc**2
        cx6 =                  ( 5.04570462e-04 -3.02466062e-02*D)*adc
        cx13=                  ( 1.68473464e-04 -5.48576870e-03*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  2.62647800e+02 +(-8.86916339e-06 +3.23008109e-06*D)*adc**2
        cy8 =  3.35086900e+00 +(-6.08839423e-07 -5.65253675e-07*D)*adc**2
        cy15=  2.38964699e-01 +(-1.41970867e-07 -1.67270348e-06*D)*adc**2
        cy24=  2.60230125e-02 +(-1.41280083e-07 -1.52144095e-06*D)*adc**2
        cy35= -1.83104099e-03 +(-5.94064581e-08 -1.16255139e-06*D)*adc**2
        cy48=  2.04091566e-03 +(-9.97915337e-09 +1.46853795e-06*D)*adc**2
        cy5 =                  (-4.99779792e-04 +3.01309493e-02*D)*adc
        cy12=                  (-1.17250417e-04 +4.18415867e-03*D)*adc
        cy1 =                  ( 2.26486871e-02 -7.33054030e-01*D)*adc
        cy4 =                  ( 9.65729204e-04 -3.01924555e-02*D)*adc
        cy9 =                  ( 1.81933708e-04 -5.03118775e-03*D)*adc

        telx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tely = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        telx = telx * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor
        tely = tely * (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) * Unknown_Scale_Factor

        adtelx,adtely = POPT2.additionaldistortion(self, telx,tely)
        telx = telx + adtelx
        tely = tely + adtely

        return telx,tely

    def focalplane2celestial_wosp(self, xt, yt, adc, m2pos3, wl):
        adtelx,adtely = POPT2.additionaldistortion(self, xt, yt)
        xt = xt - adtelx
        yt = yt - adtely

        # domain is r < 270.0 mm
        x = xt / 270.0
        y = yt / 270.0

        x = x / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor
        y = y / (1.0+(m2pos3-6.0)*wfc_scale_M2POS3_coeff) / Unknown_Scale_Factor

        D = Misc.diff_index_pbl1yold_bsl7y(self, wl)

        cx2 =  1.43748020e-02 +( 8.66331254e-11 +1.93388878e-11*D)*adc**2
        cx7 = -1.88232442e-04 +(-7.82853227e-12 +8.59386158e-12*D)*adc**2
        cx14= -6.87397752e-06 +( 1.61715989e-12 +2.00354352e-11*D)*adc**2
        cx23= -5.21300344e-07 +( 7.29677304e-13 +2.69615671e-11*D)*adc**2
        cx34=  2.39570945e-07 +(-2.82067547e-13 +2.51776050e-11*D)*adc**2
        cx47= -1.08241906e-07 +( 1.28605646e-12 -3.83590010e-11*D)*adc**2
        cx6 =                  ( 2.29520266e-08 +9.04783592e-09*D)*adc
        cx13=                  (-2.24317632e-09 +4.37694338e-08*D)*adc
        cx1 =  0.00000000e+00
        cx4 =  0.00000000e+00
        cx9 =  0.00000000e+00

        cy3 =  1.43748020e-02 +( 4.61979705e-10 +3.65991127e-12*D)*adc**2
        cy8 = -1.88232442e-04 +( 1.16038555e-11 +2.66431645e-11*D)*adc**2
        cy15= -6.87397752e-06 +( 8.12842568e-12 +4.41948555e-11*D)*adc**2
        cy24= -5.21300344e-07 +( 8.66566778e-12 +4.92555793e-11*D)*adc**2
        cy35=  2.39570945e-07 +( 1.33811721e-12 +8.16466379e-11*D)*adc**2
        cy48= -1.08241906e-07 +( 4.16136826e-13 -7.37253146e-11*D)*adc**2
        cy5 =                  (-2.32140319e-08 -3.87365915e-09*D)*adc
        cy12=                  ( 3.16034703e-10 -6.95499653e-11*D)*adc
        cy1 =                  (-1.19123300e-06 +3.85419594e-05*D)*adc
        cy4 =                  (-2.58356822e-09 +6.26527802e-09*D)*adc
        cy9 =                  (-2.32047805e-09 +2.81735279e-08*D)*adc

        tantx = \
            cx2*(x) +\
            cx7*((3*(x**2+y**2)-2)*x) +\
            cx14*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*x) +\
            cx23*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*x) +\
            cx34*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*x) +\
            cx47*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*x) +\
            cx6*(2*x*y) +\
            cx13*((4*(x**2+y**2)-3)*2*x*y) +\
            cx1*(1) +\
            cx4*(2*(x**2+y**2)-1) +\
            cx9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

        tanty = \
            cy3*(y) +\
            cy8*((3*(x**2+y**2)-2)*y) +\
            cy15*((10*(x**2+y**2)**2-12*(x**2+y**2)+3)*y) +\
            cy24*((35*(x**2+y**2)**3-60*(x**2+y**2)**2+30*(x**2+y**2)-4)*y) +\
            cy35*((126*(x**2+y**2)**4-280*(x**2+y**2)**3+210*(x**2+y**2)**2-60*(x**2+y**2)+5)*y) +\
            cy48*((462*(x**2+y**2)**5-1260*(x**2+y**2)**4+1260*(x**2+y**2)**3-560*(x**2+y**2)**2+105*(x**2+y**2)-6)*y) +\
            cy5*(x**2-y**2) +\
            cy12*((4*(x**2+y**2)-3)*(x**2-y**2)) +\
            cy1*(1)+\
            cy4*(2*(x**2+y**2)-1) +\
            cy9*((6*(x**2+y**2)**2-6*(x**2+y**2)+1))

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

        return ra_offset, de_offset, inr_offset, f, min_dist_index[f], match_obj_xdp, match_obj_ydp, match_cat_xdp, match_cat_ydp

    def makeBasis(self, tel_ra, tel_de, str_ra, str_de, t, adc, inr, m2pos3, wl):
        sep0,zpa0 = Subaru.starSepZPA(self, tel_ra, tel_de, str_ra, str_de, wl, t)
        sep1,zpa1 = Subaru.starSepZPA(self, tel_ra+d_ra, tel_de, str_ra, str_de, wl, t)
        sep2,zpa2 = Subaru.starSepZPA(self, tel_ra, tel_de+d_de, str_ra, str_de, wl, t)

        z = np.zeros_like(sep0)
        o = np.ones_like(sep0)

        xfp0_0,yfp0_0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,z)
        xfp1_0,yfp1_0 = POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,z)
        xfp2_0,yfp2_0 = POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,z)

        xfp0_1,yfp0_1 = POPT2.celestial2focalplane(self, sep0,zpa0,adc,m2pos3,wl,o)
        xfp1_1,yfp1_1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc,m2pos3,wl,o)
        xfp2_1,yfp2_1 = POPT2.celestial2focalplane(self, sep2,zpa2,adc,m2pos3,wl,o)

        xfp0 = 0.5*(xfp0_0+xfp0_1)
        xfp1 = 0.5*(xfp1_0+xfp1_1)
        xfp2 = 0.5*(xfp2_0+xfp2_1)

        yfp0 = 0.5*(yfp0_0+yfp0_1)
        yfp1 = 0.5*(yfp1_0+yfp1_1)
        yfp2 = 0.5*(yfp2_0+yfp2_1)

        xdp0,ydp0 = PFS.fp2dp(self, xfp0,yfp0,inr)
        xdp1,ydp1 = PFS.fp2dp(self, xfp1,yfp1,inr)
        xdp2,ydp2 = PFS.fp2dp(self, xfp2,yfp2,inr)
        xdp3,ydp3 = PFS.fp2dp(self, xfp0,yfp0,inr+d_inr)

        dxdpdra = xdp1-xdp0
        dydpdra = ydp1-ydp0
        dxdpdde = xdp2-xdp0
        dydpdde = ydp2-ydp0
        dxdpdinr= xdp3-xdp0
        dydpdinr= ydp3-ydp0

        return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr

    # def makeBasisGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t, adc, inr):
    #     sep0,zpa0 = Subaru.starSepZPAGaia(self, tel_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)
    #     sep1,zpa1 = Subaru.starSepZPAGaia(self, tel_ra+d_ra, tel_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)
    #     sep2,zpa2 = Subaru.starSepZPAGaia(self, tel_ra, tel_de+d_de, str_ra, str_de, str_plx, str_pmRA, str_pmDE, t)

    #     xfp0,yfp0 = POPT2.celestial2focalplane(self, sep0,zpa0,adc)
    #     xfp1,yfp1 = POPT2.celestial2focalplane(self, sep1,zpa1,adc)
    #     xfp2,yfp2 = POPT2.celestial2focalplane(self, sep2,zpa2,adc)

    #     xdp0,ydp0 = PFS.fp2dp(self, xfp0,yfp0,inr)
    #     xdp1,ydp1 = PFS.fp2dp(self, xfp1,yfp1,inr)
    #     xdp2,ydp2 = PFS.fp2dp(self, xfp2,yfp2,inr)
    #     xdp3,ydp3 = PFS.fp2dp(self, xfp0,yfp0,inr+d_inr)

    #     dxdpdra = xdp1-xdp0
    #     dydpdra = ydp1-ydp0
    #     dxdpdde = xdp2-xdp0
    #     dydpdde = ydp2-ydp0
    #     dxdpdinr= xdp3-xdp0
    #     dydpdinr= ydp3-ydp0

    #     return xdp0,ydp0, dxdpdra,dydpdra, dxdpdde,dydpdde, dxdpdinr,dydpdinr

class PFS():
    def fp2dp(self, xt, yt, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xt*np.cos(inr)+yt*np.sin(inr))+pfs_detector_zero_offset_x
        y = (xt*np.sin(inr)-yt*np.cos(inr))+pfs_detector_zero_offset_y

        return x,y

    def dp2fp(self, xc, yc, inr_deg):
        inr = np.deg2rad(inr_deg)
        x = (xc-pfs_detector_zero_offset_x)*np.cos(inr)+(yc-pfs_detector_zero_offset_y)*np.sin(inr)
        y = (xc-pfs_detector_zero_offset_x)*np.sin(inr)-(yc-pfs_detector_zero_offset_y)*np.cos(inr)

        return x,y

###
if __name__ == "__main__":
    print('basic functions for Subaru telescope, POPT2 and PFS')
