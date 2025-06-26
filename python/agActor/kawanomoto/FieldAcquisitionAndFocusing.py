# -*- coding: utf-8 -*-
import numpy as np
from pfs.utils.coordinates import Subaru_POPT2_PFS

from . import Subaru_POPT2_PFS_AG


class PFS:
    @staticmethod
    def FAinstpa(
        carray,
        darray,
        tel_ra,
        tel_de,
        dt,
        adc,
        instpa,
        m2pos3,
        wl,
        inrflag=1,
        scaleflag=0,
        maxellip=0.6,
        maxsize=20.0,
        minsize=0.92,
        maxresid=0.5,
    ):
        subaru = Subaru_POPT2_PFS.Subaru()
        inr0 = subaru.radec2inr(tel_ra, tel_de, dt)
        inr = inr0 + instpa

        pfs = Subaru_POPT2_PFS_AG.PFS()

        v_0, v_1 = pfs.makeBasis(
            tel_ra, tel_de, carray[:, 0], carray[:, 1], dt, adc, inr, m2pos3, wl
        )

        v_0 = np.insert(v_0, 2, carray[:, 2], axis=1)
        v_1 = np.insert(v_1, 2, carray[:, 2], axis=1)

        ### source filtering (substantially no filtering here)
        # maxellip =  2.0e+00
        # maxsize  =  1.0e+12
        # minsize  = -1.0e+00
        filtered_darray, v = pfs.sourceFilter(darray, maxellip, maxsize, minsize)

        ### limit number of detection
        # limit_number = 20 ### should be same size of catalog list in design ...
        # limit_flux = (np.sort(filtered_darray[:,4])[::-1])[limit_number]
        # filtered_darray = filtered_darray[np.where(filtered_darray[:,4]>=limit_flux)]

        ra_offset, de_offset, inr_offset, scale_offset, mr = pfs.RADECInRShiftA(
            filtered_darray[:, 2],
            filtered_darray[:, 3],
            filtered_darray[:, 4],
            filtered_darray[:, 7],
            v_0,
            v_1,
            inrflag,
            scaleflag,
            maxresid,
        )

        rs = mr[:, 6] ** 2 + mr[:, 7] ** 2
        rs[mr[:, 8] == 0.0] = np.nan
        md = np.nanmedian(np.sqrt(rs))

        return ra_offset, de_offset, inr_offset, scale_offset, mr, md, v

    @staticmethod
    def Focus(agarray, maxellip=0.6, maxsize=20.0, minsize=0.92):
        pfs = Subaru_POPT2_PFS_AG.PFS()

        md = pfs.agarray2momentdifference(agarray, maxellip, maxsize, minsize)

        df = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for idx in range(6):
            df[idx] = pfs.momentdifference2focuserror(md[idx])

        return df
