### ======================================================================================
#
#   APPLE CODE:
#   Module: Atmospheric boundary conditions 
#   Description: Takes entropy/T_10, gravity and returns T_int, T_eff
#   Yixian's BC: takes a0,FX,and metallicity
# 
#   Yixian Chen, Ankan Sur
#   
#   June 2023
#   
### ======================================================================================


import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from functools import reduce
import pathlib

g = np.array([[3.2,3.2,3.2,3.2],[5.6,5.6,5.6,5.6,5.6],[9.1,9.1,9.1,9.1,9.1,9.1,9.1],\
            [13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5,13.5],[18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2,18.2],\
            [22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4,22.4],[25.1,25.1,25.1,25.1,25.1,25.1],\
            [28.2,28.2,28.2,28.2,28.2,28.2]],dtype=object)
            
 
col1,col2 = np.genfromtxt(str(pathlib.Path(__file__).parent.resolve())+"/Burrows_data.dat",unpack=True)
#col1,col2 = np.genfromtxt(str(pathlib.Path(__file__).parent.resolve())+"/xgm5+.dat",unpack=True)
gravity = col2[::502]
Teffec = []
T10 = []
gr = []
g_length = round(31)
for i in range(g_length):
    Teffec.append(list(col1[i+1+i*501:502*(i+1)]))
    T10.append(list(col2[i+1+i*501:502*(i+1)]))
    gr.append(list(np.repeat(gravity[i],501)))   
    
gr = np.array(gr)
T10 = np.array(T10)
Teffec = np.array(Teffec)     

      
def Tint(grav,temp,atm_bc,planet,Y_atm,cloud):#a0='1',fx='05',metal='3.16'):
    
    
    if atm_bc=="fortney2011a": 
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
        
        px = np.array(reduce(lambda x, y : x + y, g))
        py = np.array(reduce(lambda x, y : x + y, T_10_1))
        f = np.array(reduce(lambda x, y : x + y, T_int))
        
        teff_interp = griddata((px, py), f, (grav/100.0,temp), method='linear')
        
        
        if np.isnan(teff_interp):        
            x,y = grav/100,temp
            x1 = g[5][0]
            x2 = g[6][0]
            y1 = T_10_1[5][-1]
            y2 = T_10_1[6][-1]
            t = (x-x1)/(x2-x1)
            u = (y-y1)/(y2-y1)

            te1 = interp1d(T_10_1[5],T_int[5],kind='linear', fill_value='extrapolate')
            te2 = interp1d(T_10_1[6],T_int[6],kind='linear', fill_value='extrapolate')


            return ((x2-x)*te1(y)/(x2-x1) + (x-x1)*te2(y)/(x2-x1))
        else:
        
            return teff_interp
            
            
            
        
    if atm_bc=="fortney2011b":
        T_eff_07 = np.array([[1201.6,904.7,751.2,600.5],[1202.6,  902.6,  750.7,  600.4,450.4], \
        [1202.2,  901.5,  750.5,  600.3,  450.4,  317. ,  226.4], \
        [1201.8,  901. ,  750.4,  600.3,450.4,  317. ,  226.4,  173.2,  142.7], \
        [750.3,  600.2,  450.4,  317. ,  226.4,  173.2, 142.8,  126.8,  118.9,  112.9], \
        [600.2, 450.4,  317. ,  226.4,  173.3,  142.8,  126.8,  119. ,  113.],[226.4,  173.3,142.8,  126.9,  119.1,  113.1], \
        [226.6,  173.7,  142.8,  126.9,  119.1,  113.2]],dtype=object)
            
        T_10_07 = np.array([[3471.1, 3124.9, 2894.2, 2582.4],  [3360.3, 2984.3, 2730.2, 2391.6,1966.1], \
        [3255.8, 2850.2, 2571.4, 2222.1, 1815.9, 1310.2,  838.], \
        [3166. , 2731.4, 2435.8, 2086. ,1701.6, 1231.7,  787.4,  552.4,  439.9], \
        [2330.5, 1987.6, 1617.6, 1172.6,  750.4,  527.5,419.6,  360. ,  330.1,  306.9], \
        [1918.6, 1560.9, 1133.5,  725.3,  510. ,  406.5,  348.5,  320. ,  296.9], \
        [712. ,  501.1,399.2,  342.3,  314.3,  292.3],[699.5,  493.1,  391.6,  336.1,  308.7,  286.8]],dtype=object)
        
        px = np.array(reduce(lambda x, y : x + y, g))
        py = np.array(reduce(lambda x, y : x + y, T_10_07))
        f = np.array(reduce(lambda x, y : x + y, T_eff_07))
        
        teff_interp = griddata((px, py), f, (grav/100.0,temp), method='cubic')
        
        if np.isnan(teff_interp):       
            x,y = grav/100,temp
            x1 = g[5][0]
            x2 = g[6][0]
            y1 = T_10_07[5][-1]
            y2 = T_10_07[6][-1]
            t = (x-x1)/(x2-x1)
            u = (y-y1)/(y2-y1)
            te1 = interp1d(T_10_07[5],T_eff_07[5],kind='quadratic', fill_value='extrapolate')
            te2 = interp1d(T_10_07[6],T_eff_07[6],kind='quadratic', fill_value='extrapolate')

            return ((x2-x)*te1(y)/(x2-x1) + (x-x1)*te2(y)/(x2-x1))
        else:
            return teff_interp
                    
            
    if atm_bc=="gT1fit":
        K = 1.519
        expon_teff = 1.243
        return (grav**0.167*temp/K)**(1.0/expon_teff)

        
    if atm_bc=="Burrows":
        teff_func= RegularGridInterpolator((gr[:,0],T10[0,:]),Teffec)
        return teff_func((grav,temp))
        
    if atm_bc=="Yixian":

        #### actually takes the entropy S not temp for this BC
        ys=temp 
        xg=np.log10(grav)

        if planet=="Jupiter":
            if cloud:
                folder_name = "atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_table_ammocloud_size1_FACFLX05_Y025CMS.dat"
            else:
                folder_name = "atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_FACFLX05_Y25CMS.dat"
            data2 = np.genfromtxt(folder_name,unpack=True)
            ggrid = np.transpose(data2[1].reshape(9,7))
            tingrid = np.transpose(data2[0].reshape(9,7))
            sgrid=np.transpose(data2[2].reshape(9,7))

        elif planet=="Saturn":
            if Y_atm == 0.15:
                if cloud:
                    folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y15CMS.dat"
                else:
                    folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_FACFLX05_Y15CMS.dat"
                data2 = np.genfromtxt(folder_name,unpack=True)
                ggrid = np.transpose(data2[1].reshape(9,7))
                tingrid = np.transpose(data2[0].reshape(9,7))
                sgrid=np.transpose(data2[2].reshape(9,7))

            elif Y_atm == 0.25:
                folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y25CMS.dat"
                data2 = np.genfromtxt(folder_name,unpack=True)
                ggrid = np.transpose(data2[1].reshape(9,7))
                tingrid = np.transpose(data2[0].reshape(9,7))
                sgrid=np.transpose(data2[2].reshape(9,7))
            else:
                raise Exception("Only Y_atm = 0.15 and 0.25 for now")
        else:
            raise Exception("Must specify planetary atmospheric condition-- only Jupiter and Saturn for now.")


            
        igl = np.searchsorted(ggrid[:,0],xg)-1
        igh = igl+1
        if ys<=np.min(sgrid[igl]) or ys<=np.min(sgrid[igh]):
            jl1=0
            jh1=1
            jl2=0
            jh2=1
        elif ys>=np.max(sgrid[igl]) or ys>=np.max(sgrid[igh]):
            jl1=-2
            jh1=-1
            jl2=-2
            jh2=-1
        else:
            jl1=np.where(sgrid[igl]<ys)[0][-1]
            jh1=jl1+1
            jl2=np.where(sgrid[igh]<ys)[0][-1]
            jh2=jl2+1
        
        tint1= tingrid[igl,jl1] + (ys-sgrid[igl,jl1])*(tingrid[igl,jh1]-tingrid[igl,jl1])/(sgrid[igl,jh1]-sgrid[igl,jl1])
        tint2= tingrid[igh,jl2] + (ys-sgrid[igh,jl2])*(tingrid[igh,jh2]-tingrid[igh,jl2])/(sgrid[igh,jh2]-sgrid[igh,jl2])
        tint = tint1+(xg-ggrid[:,0][igl])*(tint2-tint1)/(ggrid[:,0][igh]-ggrid[:,0][igl])

        return tint

     
def Teff(grav,temp,atm_bc,planet,Y_atm,cloud): #a0='1',fx='05',metal='3.16'):

    if atm_bc=="Yixian":
      
        if planet=="Jupiter":
            if cloud:
                folder_name = "atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_table_ammocloud_size1_FACFLX05_Y025CMS.dat"
            else:
                folder_name = "atmospheres/Jupiter_boundary/Jupiter_irradiated_3.16solar_FACFLX05_Y25CMS.dat"
            
            data2 = np.genfromtxt(folder_name,unpack=True)
            
            ggrid = np.transpose(data2[1].reshape(9,7))
            teffgrid = np.transpose(data2[3].reshape(9,7))
            sgrid=np.transpose(data2[2].reshape(9,7))
        
        elif planet=="Saturn":
            if Y_atm == 0.15:
                if cloud:
                    folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y15CMS.dat"
                
                else:
                    folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_FACFLX05_Y15CMS.dat"
                data2 = np.genfromtxt(folder_name,unpack=True)
                
                ggrid = np.transpose(data2[1].reshape(9,7))
                teffgrid = np.transpose(data2[3].reshape(9,7))
                sgrid=np.transpose(data2[2].reshape(9,7))
            elif Y_atm == 0.25:
                folder_name = "atmospheres/Saturn_boundary/Saturn_irradiated_5solar_ammocloud_10micron_FACFLX05_Y25CMS.dat"
                data2 = np.genfromtxt(folder_name,unpack=True)
                
                ggrid = np.transpose(data2[1].reshape(9,7))
                teffgrid = np.transpose(data2[3].reshape(9,7))
                sgrid = np.transpose(data2[2].reshape(9,7))
            else:
                raise Exception("Only Y_atm = 0.15 and 0.25 for now")
        else:
            raise Exception("Must specify planetary atmospheric condition-- only Jupiter and Saturn for now.")
        
        ys=temp
        xg=np.log10(grav)
   
        igl = np.searchsorted(ggrid[:,0],xg)-1
        igh = igl+1
        if xg<=np.amin(ggrid[:,0]):
            igl=0
            igh=1
        if ys<=np.min(sgrid[igl]) or ys<=np.min(sgrid[igh]):
            jl1=0
            jh1=1
            jl2=0
            jh2=1
        elif ys>=np.max(sgrid[igl]) or ys>=np.max(sgrid[igh]):
            jl1=-2
            jh1=-1
            jl2=-2
            jh2=-1
        else:
            jl1=np.where(sgrid[igl]<ys)[0][-1]
            jh1=jl1+1
            jl2=np.where(sgrid[igh]<ys)[0][-1]
            jh2=jl2+1

        teff1= teffgrid[igl,jl1] + (ys-sgrid[igl,jl1])*(teffgrid[igl,jh1]-teffgrid[igl,jl1])/(sgrid[igl,jh1]-sgrid[igl,jl1])
        teff2= teffgrid[igh,jl2] + (ys-sgrid[igh,jl2])*(teffgrid[igh,jh2]-teffgrid[igh,jl2])/(sgrid[igh,jh2]-sgrid[igh,jl2])    
        teff = teff1+(xg-ggrid[:,0][igl])*(teff2-teff1)/(ggrid[:,0][igh]-ggrid[:,0][igl])
        return teff