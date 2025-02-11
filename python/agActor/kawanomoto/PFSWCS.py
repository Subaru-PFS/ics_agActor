# -*- coding: utf-8 -*-
import numpy as np
import sys
if __name__ == '__main__':
    import Subaru_POPT2_PFS
else:
    from . import Subaru_POPT2_PFS

import astropy.io.fits as pyfits

from astropy.wcs import WCS
from astropy.wcs import Sip

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

inr_offset      = 0.00    # in degree
agcenter_radius = 241.314 # in mm
scale           = 1.00    # unknown scale parameter

class PFSWCS():
    def pfswcsTAN(self, dateobs, utstr, ra2000, dec2000, adcstr, inrstr, m2pos3, wl, instpa, agangle):
        popt2  = Subaru_POPT2_PFS.POPT2()
        subaru = Subaru_POPT2_PFS.Subaru()
        dt = dateobs+"T"+utstr+"Z"
        coord = ac.SkyCoord(ra=ra2000, dec=dec2000, unit=(au.hourangle, au.deg),
                            frame='fk5')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree
        inr = inrstr
        adc = adcstr
        az, el = subaru.radec2azel(tel_ra,tel_de,0.62,dt)

        angle0       = np.deg2rad(inr-agangle+inr_offset)
        agcenterposx = -agcenter_radius*np.cos(angle0)
        agcenterposy =  agcenter_radius*np.sin(angle0)

        sep, zpa = popt2.focalplane2celestial(agcenterposx,\
                                              agcenterposy,\
                                              adc,\
                                              inr,\
                                              el,\
                                              m2pos3,\
                                              wl,\
                                              flag)

        ra,de = subaru.starRADEC(tel_ra, tel_de, sep, zpa, wl, dt)
        ra=ra[0]
        de=de[0]

        cx2 =  1.43748020e-02
        cx7 = -1.88232442e-04
        s = 0.013/270.0
        t = cx2-2*cx7+3*cx7*(241.314/270.0)**2
        u = cx2-2*cx7+9*cx7*(241.314/270.0)**2

        angle1 = np.deg2rad(-instpa+agangle)

        CD1_1 = np.rad2deg(-s*scale*t*np.sin(angle1))
        CD1_2 = np.rad2deg( s*scale*u*np.cos(angle1))
        CD2_1 = np.rad2deg(-s*scale*t*np.cos(angle1))
        CD2_2 = np.rad2deg(-s*scale*u*np.sin(angle1))

        w = WCS(naxis=2)
        w.wcs.crpix   = [     512.5,      512.5]
        w.wcs.ctype   = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval   = [        ra,         de]
        w.wcs.cunit   = ["deg"  , "deg"  ]
        w.wcs.cd      = [[CD1_1, CD1_2],
                         [CD2_1, CD2_2]]

        return w

    def pfswcsTAN_SIP(self, dateobs, utstr, ra2000, dec2000, adcstr, inrstr, m2pos3, wl, instpa, agangle):
        #popt2  = Subaru_POPT2_PFS.POPT2()
        #subaru = Subaru_POPT2_PFS.Subaru()

        #dt = dateobs+"T"+utstr+"Z"
        coord = ac.SkyCoord(ra=ra2000, dec=dec2000, unit=(au.hourangle, au.deg),
                            frame='fk5')
        tel_ra = coord.ra.degree
        tel_de = coord.dec.degree
        #inr = inrstr
        #adc = adcstr
        #az, el = subaru.radec2azel(tel_ra,tel_de,0.62,dt)

        cx2 =  1.43748020e-02
        cx7 = -1.88232442e-04
        s = 0.013/270.0
        t = cx2-2*cx7
        u = 3*s**2*cx7/t

        dx2 =  2.62647800e+02
        dx7 =  3.35086900e+00
        sig = 0.013/270.0*t/0.014
        tau = dx2-2*dx7
        ups = 3*sig**2*dx7/tau

        angle = np.deg2rad(-instpa+agangle)

        CD1_1 = np.rad2deg(-s*scale*t*np.sin(angle))
        CD1_2 = np.rad2deg( s*scale*t*np.cos(angle))
        CD2_1 = np.rad2deg(-s*scale*t*np.cos(angle))
        CD2_2 = np.rad2deg(-s*scale*t*np.sin(angle))

        a = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, u, 0, 0],
                      [u, 0, 0, 0]])

        b = np.array([[0, 0, 0, u],
                      [0, 0, u, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])

        ap= np.array([[  0,   0,   0,   0],
                      [  0,   0,   0,   0],
                      [  0, ups,   0,   0],
                      [ups,   0,   0,   0]])

        bp= np.array([[  0,   0,   0, ups],
                      [  0,   0, ups,   0],
                      [  0,   0,   0,   0],
                      [  0,   0,   0,   0]])

        w = WCS(naxis=2)
        w.wcs.crpix = [     512.5    ,    19075.11538]
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        w.wcs.crval = [        tel_ra,         tel_de]
        w.wcs.cunit = ["deg"      , "deg"      ]
        w.wcs.cd    = [[CD1_1, CD1_2],
                       [CD2_1, CD2_2]]

        w.sip = Sip(a,b,ap,bp, [     512.5    ,    19075.11538])

        return w

###
if __name__ == "__main__":
    #popt2  = Subaru_POPT2_PFS.POPT2()
    #subaru = Subaru_POPT2_PFS.Subaru()

    pfswcs = PFSWCS()

    infile = sys.argv[1]

    inputfits   = pyfits.open(infile)
    inputdata   = inputfits[0].data
    inputheader = inputfits[0].header

    # outputdata   = inputdata/160000.0*32768
    outputdata   = inputdata/160000.0*32768

    dateobs = inputheader['DATE-OBS']
    utstr   = inputheader['UT-STR']
    ra2000  = inputheader['RA2000']
    dec2000 = inputheader['DEC2000']
    adcstr  = inputheader['ADC-STR']
    inrstr  = inputheader['INR-STR']
    m2pos3  = inputheader['M2-POS3']
    instpa  = inputheader['INST-PA']
    agangle = inputheader['AGANGLE']

    wl     = 0.62
    flag   = np.array([0])

    wcs = pfswcs.pfswcsTAN(dateobs, utstr, ra2000, dec2000, adcstr, inrstr, m2pos3, wl, instpa, agangle)
    wcsheader = wcs.to_header()
    wcsheader.remove('LATPOLE')
    wcsheader.remove('CDELT1')
    wcsheader.remove('CDELT2')
    wcsheader.remove('MJDREF')
    wcsheader.set('CD1_1', wcsheader['PC1_1'])
    wcsheader.set('CD1_2', wcsheader['PC1_2'])
    wcsheader.set('CD2_1', wcsheader['PC2_1'])
    wcsheader.set('CD2_2', wcsheader['PC2_2'])
    wcsheader.remove('PC1_1')
    wcsheader.remove('PC1_2')
    wcsheader.remove('PC2_1')
    wcsheader.remove('PC2_2')

    wcsfits   = pyfits.PrimaryHDU(outputdata.astype(np.int16),header=wcsheader)
    wcsfits.writeto('wcstantest.fits')

    wcssip = pfswcs.pfswcsTAN_SIP(dateobs, utstr, ra2000, dec2000, adcstr, inrstr, m2pos3, wl, instpa, agangle)
    wcssipheader = wcssip.to_header(relax=True)
    wcssipheader.remove('LATPOLE')
    wcssipheader.remove('CDELT1')
    wcssipheader.remove('CDELT2')
    wcssipheader.remove('MJDREF')
    wcssipheader.set('CD1_1', wcssipheader['PC1_1'])
    wcssipheader.set('CD1_2', wcssipheader['PC1_2'])
    wcssipheader.set('CD2_1', wcssipheader['PC2_1'])
    wcssipheader.set('CD2_2', wcssipheader['PC2_2'])
    wcssipheader.remove('PC1_1')
    wcssipheader.remove('PC1_2')
    wcssipheader.remove('PC2_1')
    wcssipheader.remove('PC2_2')

    wcssipfits = pyfits.PrimaryHDU(outputdata.astype(np.int16),header=wcssipheader)
    wcssipfits.writeto('wcstansiptest.fits')
