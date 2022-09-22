# -*- coding: utf-8 -*-
import numpy as np
import sys

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

if __name__ == '__main__':
    import Subaru_POPT2_PFS_AG
else:
    from . import Subaru_POPT2_PFS_AG

class PFS():
    def FA(self, carray, darray, tel_ra, tel_de, dt, adc, inr, m2pos3, wl):
        pfs  = Subaru_POPT2_PFS_AG.PFS()
        v_0, v_1 = \
            pfs.makeBasis(tel_ra, tel_de, \
                          carray[:,0], carray[:,1], \
                          dt, adc, inr, m2pos3, wl)
        v_0 = np.insert(v_0,2, carray[:,2], axis=1)
        v_1 = np.insert(v_1,2, carray[:,2], axis=1)

        maxellip = 0.6
        maxsize  =20.0
        minsize  = 1.5
        filtered_darray, v = pfs.sourceFilter(darray, maxellip, maxsize, minsize)

        ra_offset,de_offset,inr_offset, mr, md, min_dist_index_f, f = \
            pfs.RADECInRShift(filtered_darray[:,2],\
                              filtered_darray[:,3],\
                              filtered_darray[:,4],\
                              filtered_darray[:,7],\
                              v_0, v_1)

        return ra_offset,de_offset,inr_offset, mr, md, min_dist_index_f, f, v

    def Focus(self, agarray):
        pfs  = Subaru_POPT2_PFS_AG.PFS()

        maxellip = 0.6
        maxsize  =20.0
        minsize  = 1.5
        
        md = pfs.agarray2momentdifference(agarray, maxellip, maxsize, minsize)

        df = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for idx in range(6):
            df[idx] = pfs.momentdifference2focuserror(md[idx])

        return df

###
if __name__ == "__main__":
    print('### PFS field acquisition and focusing')
