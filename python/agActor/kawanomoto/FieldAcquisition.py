# -*- coding: utf-8 -*-
import numpy as np
import sys
from . import Subaru_POPT2_HSC

class PFS():
    def FA(self, stararray, detectarray, tel_ra, tel_de, dt, adc, inr, tmp=273.15, wl=0.77):
        # subaru = Subaru_POPT2_HSC.Subaru()
        popt2  = Subaru_POPT2_HSC.POPT2()
        # hsc    = Subaru_POPT2_HSC.HSC()

        str_xdp,str_ydp,dxra,dyra,dxde,dyde,dxinr,dyinr = \
            popt2.makeBasis(tel_ra, tel_de, \
                            stararray[:,0], stararray[:,1], \
                            dt, adc, inr, tmp, wl)
        filtered_detectarray = popt2.sourceFilter(detectarray)
        ra_offset,de_offset,inr_offset = \
            popt2.RADECInRShift(filtered_detectarray[:,2],\
                                filtered_detectarray[:,3],\
                                filtered_detectarray[:,4],\
                                str_xdp,str_ydp,stararray[:,2],\
                                dxra,dyra,dxde,dyde,dxinr,dyinr)

        return ra_offset,de_offset,inr_offset

###
if __name__ == "__main__":
    ###### star catalog
    ### ra[deg] dec[deg] mag
    ###
    ###### ag source catalog
    ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix]
    ###
    ###### telescope status
    ### DATE-OBS UT-STR RA2000 DEC2000 ADC-STR INR-STR ENV-TEMP WAVELENGTH
    ###

    starfile = sys.argv[1]
    agfile   = sys.argv[2]
    telstat  = sys.argv[3]

    stararray = np.loadtxt(starfile)
    agarray   = np.loadtxt(agfile)

    # subaru = Subaru_POPT2_HSC.Subaru()
    popt2  = Subaru_POPT2_HSC.POPT2()
    # hsc    = Subaru_POPT2_HSC.HSC()

    pfs = PFS()

    tel_ra, tel_de, dt, adc, inr, envtmp, wl = popt2.telStatusfromText(telstat)

    ra_offset,de_offset,inr_offset = pfs.FA(stararray, agarray, tel_ra, tel_de, dt, adc, inr , envtmp, wl)

    # print("#", ra_offset,de_offset,inr_offset+Subaru_POPT2_HSC.hsc_inr_zero_offset )
    # print("R.A. offset = %f [deg]" %( ra_offset))
    # print("Decl offset = %f [deg]" %( de_offset))
    # print("InR  offset = %f [deg]" %(inr_offset+Subaru_POPT2_HSC.hsc_inr_zero_offset))

    # print(ra_offset * 3600.0 * np.cos(np.deg2rad(tel_de)))
    # print(de_offset * 3600.0)
