import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

jup_radii = np.array([0.0923, 0.196, 0.350, 0.478, 0.584, 0.680, 0.770, 0.852,\
                0.89, 0.930, 0.952, 0.972, 0.98, 0.986])*u.Rjup.to('cm')
cond_jup = np.array([1710, 1470, 1040, 721, 465, 283, 153,\
                        59.6, 20.2, 2.78, 2.18, 1.74, 1.50, 1.30])*(u.W * u.K / u.m).to('erg*K*s^-1 / cm')

D_H_jup = np.array([0.433, 0.428, 0.436, 0.450, 0.458, 0.468, 0.481, 0.471, 0.369,\
                    0.274, 0.349, 0.410, 0.437, 0.610])*(u.mm**2 / u.s).to('cm^2/s')

D_He_jup = np.array([0.256, 0.272, 0.251, 0.254, 0.247, 0.229, 0.2, 0.179, 0.172,\
                    0.260, 0.330, 0.4, 0.430, 0.550])*(u.mm**2 / u.s).to('cm^2/s')

cond_jup_interp = interp1d(jup_radii, np.log10(cond_jup), kind='linear', fill_value='extrapolate')
DH_jup_interp = interp1d(jup_radii, D_H_jup, kind='linear', fill_value='extrapolate')
DHe_jup_interp = interp1d(jup_radii, D_He_jup, kind='linear', fill_value='extrapolate')


sat_radii = np.array([0.25, 0.3, 0.35, 0.37, 0.385, 0.395, 0.41, 0.45,\
                0.50, 0.55, 0.60, 0.65, 0.70, 0.717, 0.733, 0.75, 0.767,\
                0.783, 0.80, 0.85, 0.90, 0.95])*u.Rjup.to('cm')*0.84
cond_sat = np.array([25.3, 23.8, 14.1, 249, 245, 230, 225,\
                    177, 137, 110, 77, 51.3, 25.2, 13.1, 9.77, 5.35, 4.08,\
                    3.4, 3.11, 3.01, 3.40, 2.85])*(u.W * u.K / u.m).to('erg*K*s^-1 / cm')

D_H_sat = np.array([0.066, 0.058, 0.048, 0.327, 0.356, 0.371, 0.373, 0.391, 0.394,\
                    0.379, 0.366, 0.313, 0.252, 0.216, 0.17, 0.156, 0.151,\
                    0.177, 0.171, 0.214, 0.259, 0.732])*(u.mm**2 / u.s).to('cm^2/s')

D_He_sat = np.array([0.057, 0.049, 0.056, 0.166, 0.113, 0.176, 0.090, 0.169,\
                    0.203, 0.1, 0.175, 0.185, 0.194, 0.106, 0.069, 0.166, 0.228,\
                    0.138, 0.1, 0.259, 0.277, 1.062])*(u.mm**2 / u.s).to('cm^2/s')

cond_sat_interp = interp1d(sat_radii, np.log10(cond_sat), kind='linear', fill_value='extrapolate')
DH_sat_interp = interp1d(sat_radii, D_H_sat, kind='linear', fill_value='extrapolate')
DHe_sat_interp = interp1d(sat_radii, D_He_sat, kind='linear', fill_value='extrapolate')
