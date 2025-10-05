import os
import numpy as np
import sys
import math as mt
import re
import h5py

import mylibrary as mylib

const = {
    'k' : 1.3799999999999998e-23, # Boltzmann constant [m^2*kg*s^-2*K^-1]
    'sigma' : 5.67e-8, # stefan-boltzmann constant [Watts.m/K^4]
    'c' : 2.99e8, # speed of light
    'h' : 6.626e-34, # planck constant
}


def read_ktable(myname):
    f = h5py.File(myname,'r')
    bin_centers = np.array(f['bin_centers']) # this is in cm^-1
    bin_edges = np.array(f['bin_edges'])
    kcoeffs = np.array(f['kcoeff']) # k coefs units originally in cm^2/molecule
    myt = np.array(f['t'])
    myp = np.array(f['p']) # original P is in bar becareful
    return bin_centers,kcoeffs,myt,myp,bin_edges

def extract_kvector(myname,T,P):
    bin_centers,kcoeffs,myt,myp,bin_edges = read_ktable(myname)
    tindex = np.abs(myt-T).argmin()
    pindex = np.abs(myp-P/1e5).argmin()
    return np.flip(1e-4*(kcoeffs[pindex,tindex,:,0])),np.flip(bin_centers),np.flip(bin_edges) # convert k coef units to m^2/molecule

def initialize_RTfilenames(RTnames,runparams):
    os.chdir(runparams['paths']['exomol'])
    RTfiles = {}
    with open('datanames.txt') as input_txt:
        for line in input_txt:
            for i in np.arange(0,len(RTnames)):
                if RTnames[i]+':' in line:
                    RTfiles[RTnames[i]] = line.split(': ')[1]
                    RTfiles[RTnames[i]] = RTfiles[RTnames[i]][0:-1]
    os.chdir(runparams['paths']['home'])
    return RTfiles

def optical_depth_find(myname,T,sysparams,runparams):
    os.chdir(runparams['paths']['exomol'])
    P = mylib.total_vap_press_find(runparams['RTfiles'],T,sysparams,runparams)
    Pv = mylib.vap_press_find(myname,T,sysparams,runparams)
    if runparams['fixTbroad'] > 0:
        T = runparams['fixTbroad']
    if runparams['fixPbroad'] > 0:
        P = runparams['fixPbroad']
    kcoeffs,bin_centres,bin_edges = extract_kvector(runparams['RTfiles'][myname],T,P)
    m = sysparams['mass_mole'][myname]
    mydepth = np.zeros_like(bin_centres)
    for i in np.arange(0,len(bin_centres)):
        mydepth[i] = kcoeffs[i]*Pv/(m*sysparams['g'])
    os.chdir(runparams['paths']['home'])
    return mydepth,bin_centres,bin_edges

def total_optical_depth_find(names,T,sysparams,runparams):
    mydepth,bin_centres,bin_edges = optical_depth_find(names[0],T,sysparams,runparams)
    for i in np.arange(1,len(names)):
        mydepth2,dummy,dummy = optical_depth_find(names[i],T,sysparams,runparams)
        mydepth += mydepth2
    return mydepth,bin_centres,bin_edges

def opacity_find(names,T,sysparams,runparams):
    mydepth,bin_centres,bin_edges = total_optical_depth_find(names,T,sysparams,runparams)
    opacity = 1-np.exp(-mydepth)
    return opacity,bin_centres,bin_edges

def planck_func(wavelengths,T):
    flux = np.zeros_like(wavelengths)
    for i in np.arange(0,len(wavelengths)):
        A = 2*mt.pi*const['h']*const['c']**2/wavelengths[i]**5 # wavelength must be in metres
        B = const['h']*const['c']/(wavelengths[i]*const['k']*T)
        if B > 100:
            flux[i] = 0
        else:
            flux[i] = A/(mt.exp(B)-1)
    return mt.pi*flux

def RC_cooling(T,Tf,names,sysparams,runparams):
    mydepth,bin_centres,bin_edges = total_optical_depth_find(names,Tf,sysparams,runparams)
    wavelengths = (1e-2/bin_centres)
    planck_flux = planck_func(wavelengths,T)
    dwavelength = np.gradient(wavelengths)
    A = np.array(dwavelength)*np.array(planck_flux)
    return np.dot(mydepth,planck_flux)

def planck_int1(T,mylambda): # Using series from lambda to infinity
    k,h,c = const['k'],const['h'],const['c']
    myconst = 2*mt.pi*k**4*T**4/(h**3*c**2)
    x = h*c/(mylambda*k*T)
    n=1
    planck_int = 0
    while n <= 100:
        a = x**3/n + 3*x**2/n**2 + 6*x/n**3 + 6/n**4
        b = mt.exp(-n*x)
        planck_int = planck_int + a*b
        n = n + 1
    return myconst*planck_int

def planck_int2(T,lambda1,lambda2):
    return np.abs(planck_int1(T,lambda1)-planck_int1(T,lambda2))

def planck_int_def(T,centres,edges):
    planck_vec = np.zeros_like(centres)
    for i in np.arange(0,len(centres)-1):
        lambda1,lambda2 = edges[i],edges[i+1]
        planck_vec[i] = planck_int2(T,lambda1,lambda2)
    return planck_vec
        
def stellar_read(RTnames,sysparams,runparams):
    y,x,x_edges = optical_depth_find(RTnames[0],2000,sysparams,runparams)
    myx = (1e-2/x)
    myx_edges = (1e-2/x_edges)
    mypath = runparams['paths']['data'] + '/RTdata/' + runparams['planet_name']
    os.chdir(mypath)
    stellar_data = np.genfromtxt(runparams['planet_name']+'.txt')
    wavelengths = 1e-9*stellar_data[:,0] # original data in nm
    stellar_flux = stellar_data[:,1]
    xindex = abs(myx-max(wavelengths)).argmin()
    mainx = myx[0:xindex]
    stellar_flux2 = np.interp(mainx,wavelengths,stellar_flux)
    dmainx = np.zeros_like(mainx)
    for i in np.arange(0,len(myx_edges[0:xindex+1])-1):
        dmainx[i] = myx_edges[i+1]-myx_edges[i]
    stellarfluxint = stellar_flux2*dmainx
    myplanck = planck_int_def(sysparams['T_stellar'],mainx,myx_edges[0:xindex+1])
    myfactor = np.sum(myplanck)/np.sum(stellarfluxint)*(sysparams['r2']/sysparams['dist'])**2
    stellarfluxint2 = stellarfluxint*myfactor
    return mainx,stellarfluxint2,xindex

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MAKE GRID <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def makegrid(angle,data,lat,lon):
    map = np.zeros((np.size(lat), np.size(lon)))
    i = 0
    while i < np.size(lat):
        j = 0
        while j < np.size(lon):
            currentlat,currentlon = mt.radians(lat[i]),mt.radians(lon[j])
            currentangle = mt.acos(mt.cos(currentlat)*mt.cos(currentlon))
            if mt.degrees(currentangle) > angle[-1]:
                map[i,j] = 0
            else:
                myindex = np.abs(np.array(angle)-mt.degrees(currentangle)).argmin()
                map[i,j] = data[myindex]
            j+=1
        i+=1
    return map
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>> CALC EMISSION PER FRAME <<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def makeframe(map,centre):
    left = centre - 90 # careful, input should be in degrees and not radians
    right = centre + 90
    mylon = np.linspace(0,360,np.size(map[1,:]))
    leftindex = np.abs(np.array(mylon)-left).argmin()
    rightindex = np.abs(np.array(mylon)-right).argmin()
    if left < 0:
        left = 360 + left
        leftindex = np.abs(np.array(mylon)-left).argmin()
        leftframe = map[:,leftindex:-1]
        rightframe = map[:,0:rightindex]
        frame = np.hstack((leftframe,rightframe))
        lonrange = np.append(mylon[leftindex:-1],mylon[0:rightindex])
    elif right > 360:
        right = right - 360
        rightindex = np.abs(np.array(mylon)-right).argmin()
        leftframe = map[:,leftindex:-1]
        rightframe = map[:,0:rightindex]
        frame = np.hstack((leftframe,rightframe))
        lonrange = np.append(mylon[leftindex:-1],mylon[0:rightindex])
    else:
        frame = map[:,leftindex:rightindex]
        lonrange = mylon[leftindex:rightindex]
    return frame,lonrange
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>> CALC EMISSION PER FRAME <<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def frameint(map,lonrange):
    latrange = np.linspace(0,180,np.size(map[:,1]))
    centre = lonrange[int(np.size(lonrange)/2)]
    dlat = np.abs(latrange[2]-latrange[1])
    dlon = np.abs(lonrange[2]-lonrange[1])
    myint = 0
    i=0
    while i < np.size(latrange):
        j=0
        while j < np.size(lonrange):
            mycosterm = mt.radians(lonrange[j]-centre)
            mysinterm = mt.radians(latrange[i])
            myint = myint + map[j,i]*(mt.cos(mycosterm))*(mt.sin(mysinterm))**2*(mt.radians(1))**2
            j+=1
        i+=1
    return myint
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BLACKBODY CALC <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def blackbodycalc(T,mylambda):
    i = 0
    B = 0*np.array(T)
    h,c,k = const['h'],const['c'],const['k']
    while i < np.size(T):
        if T[i] <= 1:
            B[i] = 0
        else:
            B[i] = (2*h*c**2/mylambda**5)/(mt.exp(h*c/mylambda/k/T[i])-1)
        i+=1
    return B
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BLACKBODY CALC <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def blackbodycalc_single(T,mylambda):
    h,c,k = const['h'],const['c'],const['k']
    if T > 1:
        return (2*h*c**2/mylambda**5)/(mt.exp(h*c/mylambda/k/T)-1)
    else:
        return 0
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ECLIPSE CALC <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def eclipsecalc(angle,data,lat,lon):
    datagrid = makegrid(angle,data,lat,lon)
    dataframe,lonrange = makeframe(datagrid,0)
    mydata = frameint(dataframe,lonrange)
    return mydata
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STAR FLUX CALC <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def starfluxcalc(angle,T,mylambda,lat,lon,sysparams,runparams):
    stellarout = blackbodycalc(T*np.ones_like(angle),mylambda)
    stellarflux = eclipsecalc(angle,stellarout,lat,lon)
    return stellarflux*(sysparams['r2']/sysparams['r'])**2

def opacity_per_wavelength(T,mylambda,sysparams,runparams):
    opacity,bin_centres,bin_edges = opacity_find(runparams['RTnames'],T,sysparams,runparams)
    wavelength = 1e-2/bin_centres
    myindex = abs(wavelength-mylambda).argmin()
    return opacity[myindex]





























    