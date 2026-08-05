import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from scipy.optimize import brentq
import sys
sys.path.append('../')
from utils import const
from matplotlib import pyplot as plt

g = np.array([[3.2,3.2,3.2,3.2],[5.6,5.6,5.6,5.6,5.6],[9.1,9.1,9.1,9.1,9.1,9.1,9.1],\
            [13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5],[18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2],\
            [22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4],[25.1,25.1,25.1,25.1,25.1,25.1],\
            [28.2,28.2,28.2,28.2,28.2,28.2]],dtype=object)

T_eff_1 = np.array([[1201.6,904.7,751.3,600.6],[1201.2,902.5,750.8,600.4,450.6],\
        [1201.8,901.5,750.5,600.4,450.6,317.4,227.3],[1201.5,901.5,750.4,600.3,450.6,317.4,227.4,175.3,146.4],\
        [750.3,600.3,450.6,317.4,227.4,175.4,146.4,131.9,125.,119.7],\
        [600.3,450.6,317.4,227.4,175.4,146.5,131.9,125,119.8],\
        [227.4,175.4,146.5,132.,125.1,119.9],[227.7,176.,146.5,132.,125.1,120.]],dtype=object)   
                             
T_10_1 = np.array([[3471.1,3124.9,2894.2,2582.3],[3359.3,2984.3,2730.3,2391.7,1966.2],\
        [3255.1,2850.3,2571.4,2222.1,1816,1312.2,842.6],[3165.6,2731.4,2435.8,2086.1,1701.6,1233.5,791.8,558.7,451.4],\
        [2330.5,1987.6,1617.7,1174.5,754.4,533.5,430.9,375.8,348.9,322.9],\
        [1918.6,1561,1135.3,729.5,515.9,417.1,364.3,337.9,317.9],[716, 506.9,409.7,357.4,331.9,312.5],\
        [703.7,499.4,402.3,350.9,325.8,307.2]],dtype=object) 

T_int = np.array([[1200,900,750,600],[1200,900,750,600,450],\
        [1200,900,750,600,450,316,224],[1200,900,750,600,450,316,224,168,133],\
        [750,600,450,316,224,168,133,112,100,89],\
        [600,450,316,224,168,133,112,100,89],\
        [224,168,133,112,100,89],[224,168,133,112,100,89]],dtype=object)

names = 'g', 'teff_10', 't10_10', 'teff_07', 't10_07', 'tint'
# g_step = 12
# usecols = (0, 1, 2, 3, 4, 5)
# data = np.genfromtxt('f11_atm_jup.dat', delimiter='&', names=names, usecols=usecols)

g_grid = np.array([3.2,5.6,9.1,13.5,18.2,22.4,25.1,28.2])
tint_grid = np.array([1200,900,750,600,450,316,224,168,133,112,100,89])
npts_g = len(g_grid)
npts_tint = len(tint_grid)
t = {}
t_columns = names[1:3] # ('teff_10', 't10_10', 'teff_07', 't10_07') in the j/s case; (teff, t1, t10) in the u/n case
for t_column in t_columns:
    t[t_column] = np.zeros((npts_g, npts_tint))

for m, gval in enumerate(g_grid):
    for n, tintval in enumerate(tint_grid):
        for t_column in t_columns:
            try:
                t[t_column][m, n]=T_10_1[m][n]
            except:
                continue
                

'''
get = {}
for t_column in t_columns:
    get[t_column] = RegularGridInterpolator((g_grid, tint_grid), t[t_column])

def get_tint(g, t10, flux_level='10'):
    """
    given g (mks) and t10 (K) as arguments, does a root find to obtain the
    intrinsic temperature tint (K) and effective temperature teff (K).
    """

    if not flux_level:
        if planet in ('u', 'n'):
            t10_column = 't10'
            teff_column = 'teff'
        else:
            t10_column = 't10_10'
            teff_column = 'teff_10'
    else:
        t10_column = 't10_{:s}'.format(flux_level)
        teff_column = 'teff_{:s}'.format(flux_level)

    assert t10 > 0., 'get_tint got a negative t10 %f' % t
    def zero_me(tint):
        try:
            return t10 - get[t10_column]((g, tint))
            
        except ValueError:
            print('failed to interpolate for t10 at g = %f, tint = %f' % (g, tint))

    # for brackets on tint, consider the nodes for the two nearest values in g. take the
    # intersection of their tint values for which t10, teff are defined, and use min/max
    # of that set as our bracketing values. (avoids choosing as brackets points for which
    # t10, teff are not defined)
    if not min(g_grid) <= g <= max(g_grid):
        raise ValueError('g value %f outside bounds of f11 atmosphere table for %s' % (g, planet))
    g_bin = np.where(g_grid - g > 0)[0][0]
    g_lo, g_hi = g_grid[g_bin - 1], g_grid[g_bin]
    
    data_g_lo = data[data['g'] == g_lo]
        
    data_g_lo = data_g_lo[data_g_lo[t10_column] > 0]
    min_tint_g_lo = min(data_g_lo['tint'])
    max_tint_g_lo = max(data_g_lo['tint'])

    data_g_hi = data[data['g'] == g_hi]
    data_g_hi = data_g_hi[data_g_hi[t10_column] > 0]
    min_tint_g_hi = min(data_g_hi['tint'])
    max_tint_g_hi = max(data_g_hi['tint'])

#     min_tint_g_lo = min(T_int[g_bin-1])
#     max_tint_g_lo = max(T_int[g_bin-1])
# 
#     min_tint_g_hi = min(T_int[g_bin])
#     max_tint_g_hi = max(T_int[g_bin])
#     
#     min_tint = max(min_tint_g_lo, min_tint_g_hi)
#     max_tint = min(max_tint_g_lo, max_tint_g_hi)

    tint = brentq(zero_me, min_tint, max_tint)
#     teff = float(get[teff_column]((g, tint)))
#     if np.isnan(teff):
#         _ = np.linspace(min_tint, max_tint)
#         while np.isnan(teff):
#             _ = np.delete(_, -1)
#             teff = splev(tint, splrep(_, get[teff_column]((g, _))), ext=2)
    return tint
    
'''

xgrid = g_grid
ygrid = T_10_1
zgrid = T_int

def get_tint(x,y):
    
    if x==xgrid[0]:
        igl=0
        igh=1
    else:
        igl = np.searchsorted(xgrid,x)-1
        igh = igl+1
        
    if x>=np.amax(xgrid):

        igl=-2
        igh=-1
    if y<=np.min(ygrid[igl]) or y<=np.min(ygrid[igh]):

        jl1=-1
        jh1=-2
        jl2=-1
        jh2=-2
    elif y>=np.max(ygrid[igl]) or y>=np.max(ygrid[igh]):

        jl1=0
        jh1=1
        jl2=0
        jh2=1
    else:

        jh1=np.where(np.array(ygrid[igl])<y)[0][0]
        jl1=jh1-1
        jh2=np.where(np.array(ygrid[igh])<y)[0][0]
        jl2=jh2-1
        
    
    teff1= zgrid[igl][jl1] + (y-ygrid[igl][jl1])*(zgrid[igl][jh1]-zgrid[igl][jl1])/(ygrid[igl][jh1]-ygrid[igl][jl1])
    #print ("teff1=",teff1)
    teff2= zgrid[igh][jl2] + (y-ygrid[igh][jl2])*(zgrid[igh][jh2]-zgrid[igh][jl2])/(ygrid[igh][jh2]-ygrid[igh][jl2])
    #print ("teff2=",teff2)
    teff = teff1+(x-xgrid[igl])*(teff2-teff1)/(xgrid[igh]-xgrid[igl])
    return teff