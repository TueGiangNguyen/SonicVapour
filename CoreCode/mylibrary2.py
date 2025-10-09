# code to do data handling

import re
import math as mt
import numpy as np
import os

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CONSTANTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
const = {
    'k' : 1.3799999999999998e-23, # Boltzmann constant [m^2*kg*s^-2*K^-1]
    'sigma' : 5.67e-8, # stefan-boltzmann constant [Watts.m/K^4]
    'c' : 2.99e8, # speed of light
    'h' : 6.626e-34, # planck constant
    'amu': 1.16054e-27, # atomic mass units [kg]
    'avo': 6.022e23 # avogadro's constant
}

mole_mass = {
    'O2': 32.00, 'O': 16.00, 'Si': 28.09, 'Si2': 56.18, 'Si3': 84.27,
    'SiO': 44.09, 'SiO2': 60.08, 'Al': 26.98, 'Al2': 53.96, 'AlO': 42.98,
    'Al2O': 69.96, 'AlO2': 58.98, 'Al2O2': 85.96, 'Ti': 47.87, 'TiO': 63.87,
    'TiO2': 79.87, 'Fe': 55.85, 'FeO': 71.85, 'Mg': 24.31, 'Mg2': 48.62,
    'MgO': 40.31, 'Ca': 40.08, 'Ca2': 80.16, 'CaO': 56.08, 'Na': 22.99,
    'Na2': 45.98, 'NaO': 38.99, 'K': 39.10, 'K2': 78.20, 'KO': 55.10
}

Cp_values = {
    'SiO': 851, 'O2': 1088, 'O': 1058, 'Na': 904, 'SiO2': 1428, 'MgO': 1375,
    'K': 895, 'CaO': 1053, 'TiO': 639, 'AlO': 768
}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> READING INPUT FILE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def read_input(input_name,home):
    os.chdir(home)
    regexp1 = re.compile(r'dtheta =.*?([0-9.-]+)')
    regexp2 = re.compile(r'RT factor =.*?([0-9.-]+)')
    regexp3 = re.compile(r'La factor =.*?([0-9.-]+)')
    regexp4 = re.compile(r'GeoF =.*?([0-9.-]+)')
    with open(input_name) as input_txt:
        for line in input_txt:
            match = regexp1.match(line)
            if match:
                dtheta = float(match.group(1))
            match = regexp2.match(line)
            if match:
                RT_factor = float(match.group(1))
            match = regexp3.match(line)
            if match:
                La_factor = float(match.group(1))
            if 'RUN NAME' in line:
                dummyline = line
                dummystr = ': '
                run_name = dummyline.split(dummystr)[1]
                run_name = run_name[0:-1]
            if 'planet' in line:
                dummyline = line
                dummystr = ': '
                planet_name = dummyline.split(dummystr)[1]
                planet_name = planet_name[0:-1]
            match = regexp4.match(line)
            if match:
                geof = float(match.group(1))
    runparams = {}
    dummynames = ['run_name','planet_name','dtheta','RT_factor','La_factor','geof']
    for i in np.arange(0,len(dummynames)):
        runparams[dummynames[i]] = locals()[dummynames[i]]

    paths = {}
    paths['home'] = home
    paths['planetparams'] = home+'/data/planetparams'
    paths['mycomp'] = home+'/data/mycomps'
    paths['data'] = home+'/data'
    paths['exomol'] = home+'/data/exomol'
    runparams['paths'] = paths
    
    return runparams

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>> READ PLANET AND STELLAR INFO <<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def read_parameterdata(runparams):
    planet_name = runparams['planet_name']
    run_name = runparams['run_name']
    os.chdir(runparams['paths']['planetparams'])
    regexp1 = re.compile(r'effective stellar temperature =.*?([0-9.-]+)')
    regexp2 = re.compile(r'planetary radius =.*?([0-9.-]+)')
    regexp3 = re.compile(r'stellar radius =.*?([0-9.-]+)')
    regexp4 = re.compile(r'gravity at the surface =.*?([0-9.-]+)')
    regexp5 = re.compile(r'distance between planet and star =.*?([0-9.-]+)')
    with open(planet_name+'.txt') as input_txt:
        for line in input_txt:
            match = regexp1.match(line)
            if match:
                T_stellar = float(match.group(1))
            match = regexp2.match(line)
            if match:
                r = float(match.group(1))
            match = regexp3.match(line)
            if match:
                r2 = float(match.group(1))
            match = regexp4.match(line)
            if match:
                g = float(match.group(1))
            match = regexp5.match(line)
            if match:
                dist = float(match.group(1))
            # if 'stellar directory' in line:
            #     dummyline = line
            #     dummystr = ': '
            #     path_stellar = dummyline.split(dummystr)[1]
            #     # path_stellar = path_stellar[0:-1]
            #     path_stellar = home+path_stellar
    theta1 = mt.acos((r+r2)/dist)
    theta2 = mt.acos((r-r2)/dist)
    L = const['sigma']*T_stellar**4
    sysparams = {}
    dummynames = ['T_stellar','r','r2','g','dist','theta1','theta2','L']
    for i in np.arange(0,len(dummynames)):
        sysparams[dummynames[i]] = locals()[dummynames[i]]  
    os.chdir(runparams['paths']['home'])
    return sysparams

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> READ RAD TRANS DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def read_RT(runparams):
    os.chdir(runparams['paths']['data']+'/RTdata/RC_cooling')
    Tspace = np.genfromtxt('Tspace.txt')
    molespace = np.genfromtxt('molespace.txt')
    RCtable = np.genfromtxt('BSE_RC.txt')
    os.chdir(runparams['paths']['data']+'/RTdata/{s}'.format(s=runparams['planet_name']))
    stelabsvec = np.genfromtxt('BSE_stellar.txt')
    #stelabsvec = stelabsvec[:,1:3]
    return Tspace,molespace,RCtable,stelabsvec

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>> READ PRESSURE DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def vap_press_coef_make(headers,mycomp):
    Pv_coefs = {}
    oneT = 1/mycomp[0,:]
    for i in np.arange(1,len(headers)):
        dummyP = np.log(10**mycomp[i,:])
        [dummym,dummyb] = np.polyfit(oneT,dummyP,1)
        Pv_coefs['m_'+headers[i]] = dummym
        Pv_coefs['b_'+headers[i]] = dummyb
    return Pv_coefs

def read_Pv(filename,runparams):
    os.chdir(runparams['paths']['mycomp'])
    mycomp = np.genfromtxt(filename)
    headers = ['Temp.', 'O2', 'O', 'Si', 'Si2', 'Si3', 'SiO', 'SiO2', 'Al', 'Al2', 'AlO', 'Al2O', 'AlO2', 'Al2O2', 'Ti', 
           'TiO', 'TiO2', 'Fe', 'FeO', 'Mg', 'Mg2', 'MgO', 'Ca', 'Ca2', 'CaO', 'Na', 'Na2', 'NaO', 'K', 'K2', 'KO', 'total']
    names = headers[1:-1]
    Pv_coefs = vap_press_coef_make(headers,mycomp)
    os.chdir(runparams['paths']['home'])
    return names,Pv_coefs

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> READ CP DATA <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def read_Cp(names,runparams):
    mass_mole = {}
    Cp_vector = np.zeros(len(names))
    mylist = list(Cp_values)
    for i in np.arange(0,len(names)):
        mass_mole[names[i]] = mole_mass[names[i]]/1000/const['avo']
        for j in np.arange(0,len(mylist)):
            if names[i] == mylist[j]:
                Cp_vector[i] = Cp_values[names[i]]
    mass_mole_vec = np.zeros_like(Cp_vector)
    for i in np.arange(0,len(names)):
        mass_mole_vec[i] = float(mass_mole[names[i]])
    runparams['mass_mole'] = mass_mole_vec
    return mass_mole,Cp_vector,runparams

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>> FIND INDEX OF 2 DIVERGING ARRAYS <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def find_index_array(array1,array2,tolerance):
    n = 0
    minlength = min(len(array1),len(array2))
    while n < minlength:
        if abs(array1[n]-array2[n]) > tolerance:
            break
        n+=1
    return n

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>> FIND INDEX OF 2 DIVERGING ARRAYS <<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<














