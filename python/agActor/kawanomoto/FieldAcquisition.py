# -*- coding: utf-8 -*-
import numpy as np
import sys
if __name__ == '__main__':
    import Subaru_POPT2_PFS
else:
    from . import Subaru_POPT2_PFS

class PFS():
    def FA(self, catalogarray, detectarray, tel_ra, tel_de, dt, adc, inr, m2pos3, wl, verbose=False):
        # subaru = Subaru_POPT2_PFS.Subaru()
        popt2  = Subaru_POPT2_PFS.POPT2()
        # hsc    = Subaru_POPT2_PFS.HSC()

        str_xdp,str_ydp,dxra,dyra,dxde,dyde,dxinr,dyinr = \
            popt2.makeBasis(tel_ra, tel_de, \
                            catalogarray[:,0], catalogarray[:,1], \
                            dt, adc, inr, m2pos3, wl)
        v = popt2.sourceFilter(detectarray)
        filtered_detectarray = detectarray[v]
        ra_offset,de_offset,inr_offset,f,min_dist_index_f,obj_xdp,obj_ydp,cat_xdp,cat_ydp = \
            popt2.RADECInRShift(filtered_detectarray[:,2],\
                                filtered_detectarray[:,3],\
                                filtered_detectarray[:,4],\
                                str_xdp,str_ydp,catalogarray[:,2],\
                                dxra,dyra,dxde,dyde,dxinr,dyinr)

        return (ra_offset,de_offset,inr_offset,v,f,min_dist_index_f,obj_xdp,obj_ydp,cat_xdp,cat_ydp) if verbose else (ra_offset,de_offset,inr_offset)

###
if __name__ == "__main__":
    ###### star catalog
    ### ra[deg] dec[deg] mag
    ###
    ###### ag source catalog
    ### ccdid objectid xcent[mm] ycent[mm] flx[counts] semimajor[pix] semiminor[pix] flag[0 or 1]
    ###
    ###### telescope status
    ### DATE-OBS UT-STR RA2000 DEC2000 ADC-STR INR-STR M2-POS3 WAVELEN
    ###

    starfile = sys.argv[1]
    agfile   = sys.argv[2]
    telstat  = sys.argv[3]

    stararray = np.loadtxt(starfile)
    agarray   = np.loadtxt(agfile)

    popt2  = Subaru_POPT2_PFS.POPT2()

    pfs = PFS()

    # tel_ra, tel_de, dt, adc, inr, m2pos3, wl = popt2.telStatusfromText(telstat)
    # ra_offset,de_offset,inr_offset = pfs.FA(stararray, agarray, tel_ra, tel_de, dt, adc, inr, m2pos3, wl)
    # print("#", ra_offset,de_offset,inr_offset+Subaru_POPT2_PFS.pfs_inr_zero_offset )
    # print("R.A. offset = %f [deg]" %( ra_offset))
    # print("Decl offset = %f [deg]" %( de_offset))
    # print("InR  offset = %f [deg]" %(inr_offset+Subaru_POPT2_PFS.pfs_inr_zero_offset))

    # print(ra_offset * 3600.0 * np.cos(np.deg2rad(tel_de)))
    # print(de_offset * 3600.0)
