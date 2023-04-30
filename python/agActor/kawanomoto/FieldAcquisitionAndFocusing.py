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

### Subaru location
sbr_lat =   +19.8225
sbr_lon =  +204.523972222
sbr_hei = +4163.0
Lsbr = ac.EarthLocation(lat=sbr_lat,lon=sbr_lon,height=sbr_hei)

### misc.
sbr_press  = 620.0

### Telescope mount model parameters (2023/04/29)
PFS_a0 =  0.314306
PFS_a1 =  0.038765
PFS_a2 = -0.006086
PFS_a3 = -0.001102
PFS_a4 =  0.004851
PFS_a5 =  0.006647
PFS_a6 = -0.006588

class PFS():
    def FA(self, carray, darray, tel_ra, tel_de, dt, adc, inr, m2pos3, wl, inrflag=1, scaleflag=0):
        tel_coord = ac.SkyCoord(ra=tel_ra, dec=tel_de, unit=(au.deg, au.deg), frame='fk5')
        frame_subaru = ac.AltAz(obstime  = dt, location = Lsbr,\
                                pressure = sbr_press*au.hPa, obswl = wl*au.micron)
        tel_altaz = tel_coord.transform_to(frame_subaru)
        az = tel_altaz.az.degree+180.0
        el = tel_altaz.alt.degree

        PFS_beta = (np.arctan2((PFS_a3-(PFS_a4+PFS_a5)*np.cos(az/180*np.pi)),(PFS_a2+(PFS_a4+PFS_a5)*np.sin(az/180*np.pi))))*180/np.pi+180.0
        PFS_z = np.sqrt((PFS_a3-(PFS_a4+PFS_a5)*np.cos(az/180*np.pi))**2+(PFS_a2+(PFS_a4+PFS_a5)*np.sin(az/180*np.pi))**2)
        PFS_eps = np.arctan2((np.sin(PFS_z/180*np.pi)*(np.sin(az/180*np.pi-PFS_beta/180*np.pi))) , (np.cos(PFS_z/180*np.pi)*np.cos(el/180*np.pi)-np.sin(PFS_z/180*np.pi)*np.sin(el/180*np.pi)*np.cos(az/180*np.pi-PFS_beta/180*np.pi)))/np.pi*180
        # print("#",PFS_eps, az, el)
        pfs  = Subaru_POPT2_PFS_AG.PFS()
        v_0, v_1 = \
            pfs.makeBasis(tel_ra, tel_de, \
                          carray[:,0], carray[:,1], \
                          dt, adc, inr + PFS_eps, m2pos3, wl)

        v_0 = np.insert(v_0,2, carray[:,2], axis=1)
        v_1 = np.insert(v_1,2, carray[:,2], axis=1)

        maxellip = 0.6
        maxsize  =20.0
        minsize  = 1.5
        filtered_darray, v = pfs.sourceFilter(darray, maxellip, maxsize, minsize, flag_mask=7)  # ignore "fwhm not converged" flag

        ra_offset,de_offset,inr_offset, scale_offset, mr, min_dist_index_f, f = \
            pfs.RADECInRShiftA(filtered_darray[:,2],\
                                   filtered_darray[:,3],\
                                   filtered_darray[:,4],\
                                   filtered_darray[:,7],\
                                   v_0, v_1,\
                                   inrflag, scaleflag)
        rs = mr[:,6]**2+mr[:,7]**2
        rs[mr[:,8]==0.0]=np.nan
        md = np.nanmedian(np.sqrt(rs))

        return ra_offset,de_offset,inr_offset, scale_offset, mr, md, min_dist_index_f, f, v

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
