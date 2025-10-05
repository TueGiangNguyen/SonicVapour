### FUNCTION LIBRARY complex composition lava planet hydrodynamics

import re
import math as mt
import numpy as np

import mylibrary2 as mylib2
import mylibraryRT as mylibRT

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CONSTANTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
const = {
    'k' : 1.3799999999999998e-23, # Boltzmann constant [m^2*kg*s^-2*K^-1]
    'sigma' : 5.67e-8, # stefan-boltzmann constant [Watts.m/K^4]
    'c' : 2.99e8, # speed of light
    'h' : 6.626e-34, # planck constant
    'La' :   1e6, # latent heat of vaporization of SiO [J / kg]
    'amu': 1.16054e-27, # atomic mass units [kg]
    'avo': 6.022e23 # avogadro's constant
}

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> INSTELLATION CALCULATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def L_calc(theta,sysparams): # calculate the instellation at a given angular distance [radians]
    r,r2,d,theta1,theta2 = sysparams['r'],sysparams['r2'],sysparams['dist'],sysparams['theta1'],sysparams['theta2']
    L = sysparams['L']
    geof = sysparams['geof']
    d2 = mt.sqrt(d**2+r**2-2*d*r*mt.cos(theta))
    if theta <= theta1:
        return L*((d*mt.cos(theta)-r)*r2**2/d2**3) + geof
    if theta > theta1 and theta < theta2:
        alpha = mt.acos((d*mt.cos(theta)-r)/d2)
        phi = mt.asin(r2/d2)
        beta = mt.pi/2 - alpha
        delta1 = mt.acos(mt.cos(phi)/mt.cos(beta))
        delta2 = mt.acos(mt.tan(beta)/mt.tan(phi))
        term1 = mt.sin(phi)**2*mt.cos(alpha)/mt.pi*(mt.pi-delta2+mt.sin(delta2)*mt.cos(delta2))
        term2 = (delta1-mt.sin(delta1)*mt.cos(delta1))/mt.pi
        return L*(term1+term2) + geof
    if theta >= theta2:
        return geof


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>> SAT VAPOUR CALCULATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
    
def vap_press_find(name,T,sysparams,runparams):
    Pv_coefs = sysparams['Pv_coefs']
    B = Pv_coefs['m_'+name]
    A = mt.exp(Pv_coefs['b_'+name])
    return A*mt.exp(B*(1/T))*1e5 # bars -> Pa
        
def total_vap_press_find(names,T,sysparams,runparams):
    totalpress = 0
    names = runparams['names']
    for i in np.arange(0,len(names)):
        mypress = vap_press_find(names[i],T,sysparams,runparams)
        totalpress += mypress
    return totalpress

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> UPDATE COMPOSITION PARAMETERS <<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def update_comp(T,sysparams,runparams):
    names = list(sysparams['mass_mole'])
    mass_mole = runparams['mass_mole']
    Cp_vector = runparams['Cp_vector']
    Pv_coef = sysparams['Pv_coefs']
    P = np.zeros(len(names))
    for i in np.arange(0,len(names)):
        P[i] = vap_press_find(names[i],T,sysparams,runparams)
    mass_frac = P/np.sum(P)
    sysparams['m'] = np.dot(mass_frac,mass_mole)
    sysparams['R'] = const['k']/sysparams['m']
    sysparams['Cp'] = np.dot(mass_frac,Cp_vector)
    if runparams['TP'] == 'isothermal':
        sysparams['Beta'] = sysparams['R']/sysparams['Cp']
    else:
        sysparams['Beta'] = sysparams['R']/(sysparams['R']+sysparams['Cp'])
    return sysparams,mass_frac

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>> SURFACE TEMPERATURE CALCULATION <<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def surf_temp_calc_noRT(theta,sysparams):
    return (L_calc(theta,sysparams)/const['sigma'])**(1/4)

def surf_temp_calc(theta,T,P,E,sysparams,runparams):
    QRC = total_RC_cooling(T,P,sysparams)
    Qstellar = total_stellar_absorb(P,sysparams)
    Qlatent = 0
    if E > 0:
        Qlatent = const['La']*sysparams['m']*E
    totalflux = (1-runparams['RT_factor']*Qstellar)*L_calc(theta,sysparams) + runparams['RT_factor']*QRC-runparams['La_factor']*Qlatent
    return (totalflux/const['sigma'])**(1/4)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RT PORTION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def stellar_absorb(P,sysparams):
    molespace = sysparams['stelabsvec'][:,2]
    myvector = sysparams['stelabsvec'][:,3]
    molecules = P/sysparams['m']/sysparams['g']
    moleindex = np.abs(molespace-molecules).argmin() # P goes from big -> small
    if molecules < molespace[moleindex]: # careful, less means to the right
        moleindex1,moleindex2 = moleindex,moleindex+1
    if molecules > molespace[moleindex]: # index to the left
        moleindex1,moleindex2 = moleindex-1,moleindex
    if molespace[moleindex]==molecules:
        moleindex1,molindex2 = moleindex,moleindex
        myf = myvector[moleindex]
    if moleindex == 0 or moleindex == np.size(molespace)-1:
        moleindex1,moleindex2 = moleindex,moleindex
        return myvector[moleindex]
    mol1,mol2 = molespace[moleindex1],molespace[moleindex2]
    f1,f2 = myvector[moleindex1],myvector[moleindex2]
    deltaf = f2-f1
    deltamol = mol2-mol1
    c = f2 - (deltaf/deltamol)*mol2
    myf = (deltaf/deltamol)*molecules + c
    return myf

def find_nearest_points(T,P,sysparams):
    mole = P/sysparams['m']/sysparams['g']
    moleindex = np.abs(sysparams['molespace']-mole).argmin() # mole goes from big -> small
    Tindex = np.abs(sysparams['Tspace']-T).argmin() # T goes from big -> small
    if mole < sysparams['molespace'][moleindex]: # careful, less means to the right
        moleindex1,moleindex2 = moleindex,moleindex+1
    if mole > sysparams['molespace'][moleindex]: # index to the left
        moleindex1,moleindex2 = moleindex-1,moleindex
    if sysparams['molespace'][moleindex]==mole:
        moleindex1,moleindex2 = moleindex,moleindex
    if moleindex == 0 or moleindex == np.size(sysparams['molespace'])-1:
        moleindex1,moleindex2 = moleindex,moleindex
    if T < sysparams['Tspace'][Tindex]: # careful, less means to the right
        Tindex1,Tindex2 = Tindex,Tindex+1
    if T > sysparams['Tspace'][Tindex]: # index to the left
        Tindex1,Tindex2 = Tindex-1,Tindex
    if sysparams['Tspace'][Tindex]==T:
        Tindex1,Tindex2 = Tindex,Tindex
    if Tindex == 0 or Tindex == np.size(sysparams['Tspace']) - 1:
        Tindex1,Tindex2 = Tindex,Tindex
    return moleindex1,moleindex2,Tindex1,Tindex2

def RC_cooling(T,P,sysparams):
    mole = P/sysparams['m']/sysparams['g']
    mytable = sysparams['RCtable']
    moleindex1,moleindex2,Tindex1,Tindex2 = find_nearest_points(T,P,sysparams)
    if moleindex1 == moleindex2 and Tindex1 == Tindex2:
        return mytable[Tindex1,moleindex1]
    mole1,mole2 = sysparams['molespace'][moleindex1],sysparams['molespace'][moleindex2]
    T1,T2 = sysparams['Tspace'][Tindex1],sysparams['Tspace'][Tindex2]
    f11,f12 = mytable[Tindex1,moleindex1],mytable[Tindex1,moleindex2]
    f21,f22 = mytable[Tindex2,moleindex1],mytable[Tindex2,moleindex2]
    if moleindex1 == moleindex2: # mole is constant
        deltaf = f11 - f21 # f(T1,mole) - f(T2,mole)
        deltax = T1 - T2
        c = f11 - (deltaf/deltax)*T1
        myf = (deltaf/deltax)*T + c
        return myf
    if Tindex1 == Tindex2: # T is constant
        deltaf = f11 - f12 # f(T,mole1) - f(T,mole2)
        deltax = mole1 - mole2
        c = f11 - (deltaf/deltax)*mole1
        myf = (deltaf/deltax)*mole + c
        return myf
    if moleindex1 != moleindex2 and Tindex1 != Tindex2: # make a plane from 4 points
        vector1 = [T1-T2,mole1-mole2,f11-f22]
        vector2 = [T2-T1,mole1-mole2,f21-f12]
        norm = np.cross(vector1,vector2) # cross-product to get normal vector
        D = np.dot(norm,[T1,mole1,f11]) # solve for D with single point to get equation of plane
        myf = (D - norm[0]*T - norm[1]*mole)/norm[2] # isolate to solve "z"
    return myf

def total_RC_cooling(T,P,sysparams):
    molecules = P/sysparams['g']/sysparams['m']
    if molecules > max(sysparams['molespace']):
        return np.max(sysparams['RCtable'])*const['sigma']*T**4
    if molecules < min(sysparams['molespace']):
        return np.min(sysparams['RCtable'])*const['sigma']*T**4
    else:
        return const['sigma']*T**4*RC_cooling(T,P,sysparams)

def total_stellar_absorb(P,sysparams):
    molecules = P/sysparams['g']/sysparams['m']
    molespace = sysparams['stelabsvec'][:,2]
    if molecules<min(molespace):
        return min(sysparams['stelabsvec'][:,3])
    if molecules>max(molespace):
        return max(sysparams['stelabsvec'][:,3])
    else:
        return stellar_absorb(P,sysparams)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>> THERMODYNAMICS CALCULATION <<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def w_calc(V,P,Tf,m,E,R,H,Pv): # find transfer coefficients
    rho = Pv/(R*Tf)
    Ve = m*E/rho
    V_fric2 = 1E-6
    eta = 1.8E-5*((Tf/291)**(1.5))*((291+120)/(Tf+120))
    dummycondition = 0
    dummycounter = 0
    while dummycondition == 0:
        V_fric = V/(2.5*(mt.log10(abs(9*V_fric2*(H/2)*rho/eta))))
        if abs(V_fric-V_fric2)<=1E-6:
            dummycondition = 1
        elif dummycounter > 1000:
            dummycondition = 1
            # print('warning: non-convergent friction velocity')
            # print('V,Vfric,Vfric2 =',V,V_fric,V_fric2)
        else:
            V_fric2 = V_fric
            dummycounter = dummycounter + 1
    if V == 0:
        Vd = 0
    else:
        Vd = V_fric**2/V
    if Ve <= 0:
        ws = 2.*Vd**2./(-Ve+2.*Vd)
    else:
        ws = (Ve**2+2*Vd*Ve+2*Vd**2)/(Ve+2*Vd)

    if Ve < 0:
        wa = (Ve**2-2*Vd*Ve+2*Vd**2)/(-Ve+2*Vd)
    else:
        wa = 2.*Vd**2./(Ve+2.*Vd)
    return ws,wa,Ve,Vd,V_fric

def Qsens_calc(y,x,Tf,sysparams,runparams): # sensible heating calculation
    R = sysparams['R']
    g = sysparams['g']
    m = sysparams['m']
    Cp = sysparams['Cp']
    Beta = sysparams['Beta']
    k = const['k']
    V = y[0]
    P = y[1]
    T = y[2]
    Pv = total_vap_press_find(runparams['names'],Tf,sysparams,runparams)
    rho_s = Pv/(R*Tf)
    mu_s = mt.sqrt(k*Tf/m)
    E = 1*(Pv-P)/(mt.sqrt(2*mt.pi*R*Tf)*m)
    H = k*T/(m*g)
    ws,wa,Ve,Vd,V_fric = w_calc(V,P,Tf,m,E,R,H,Pv)
    qs = Cp*Tf
    qa = V**2/2 + Cp*T
    tau = rho_s*(-V*wa)
    Q1 = rho_s*(ws*qs-wa*qa)
    return Q1,tau

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>> HYDRODYNAMICAL PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def VPTsolve(f1,f2,f3,Q,E,theta,sysparams,runparams): # solve for state variables
    Beta = sysparams['Beta']
    Cp = sysparams['Cp']
    radical = f2**2 - 2.*Beta*(2.-Beta)*f1*f3
    status = 1
    if radical <= 0:
        status = 0
        return 0,0,0,0,status
    if runparams['flow'] == 'subsonic':
        V = (f2-mt.sqrt(radical))/(f1*(2-Beta)) # only difference between supersonic and subsonic
    if runparams['flow'] == 'transonic':
        V = (f2+mt.sqrt(radical))/(f1*(2-Beta)) # only difference between supersonic and subsonic
    if V<=0:
        status = 2
        V = 0
        return 0,0,0,0,status
    P = f1/V
    if P < 0:
        status = 2
        return 0,0,0,0,status
    T = (f2/P-V**2)/(Beta*Cp)
    if T < 0:
        status = 2
        return 0,0,0,0,status
    dfmass,dfmom,dfenergy,Q,Tf,E = dy_calc([V,P,T,Q,E],theta,sysparams,runparams)
    return V,P,T,Q,status

def y_calc(y,x,sysparams): # solve for LHS of ODE
    g = sysparams['g']
    r = sysparams['r']
    m = sysparams['m']
    k = const['k']
    Cp = sysparams['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
    V = y[0]
    P = y[1]
    T = y[2]
    fmass = V*P*mt.sin(x)
    fmom = (V**2 + Beta*Cp*T)*P*mt.sin(x)
    fenergy = ((V**2)/2+Cp*T)*V*P*mt.sin(x)
    return fmass,fmom,fenergy

def dy_calc(y,x,sysparams,runparams): # solve for RHS of ODE
    g = sysparams['g']
    r = sysparams['r']
    m = sysparams['m']
    k = const['k']
    Cp = sysparams['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
    names = runparams['names']
    Pv_coefs = sysparams['Pv_coefs']
    V = y[0]
    P = y[1]
    T = y[2]
    Q = y[3]
    E = y[4]
    if runparams['RT_factor'] == 0:
        Tf = surf_temp_calc_noRT(x,sysparams)
    else:
        Tf = surf_temp_calc(x,T,P,E,sysparams,runparams)
    Qsens,tau = Qsens_calc([V,P,T],x,Tf,sysparams,runparams)
    Pv = total_vap_press_find(runparams['names'],Tf,sysparams,runparams)
    E = (Pv-P)/(mt.sqrt(2*mt.pi*R*Tf)*m)
    if runparams['RT_factor'] == 0:
        Qstellar,QRC,Qsurf = 0,0,0
    else:
        Qstellar = L_calc(x,sysparams)*total_stellar_absorb(P,sysparams)
        QRC = total_RC_cooling(T,P,sysparams)
        Qsurf = total_RC_cooling(Tf,P,sysparams)
    Qcloud = 0
    if E < 0:
        Qcloud = -m*const['La']*E
    Q = Qsens + runparams['RT_factor']*(Qstellar + Qsurf - 2*QRC) + runparams['La_factor']*Qcloud
    dfmass = m*E*g*r*mt.sin(x)
    dfmom = (Beta*Cp*T*P*mt.cos(x))+tau*g*r*mt.sin(x)
    dfenergy = Q*g*r*mt.sin(x)
    return dfmass,dfmom,dfenergy,Q,Tf,E

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>> SINGLE RUN WITH BOUNDARY CONDITIONS <<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def systemtest(y0,P,xstart,sysparams,runparams): # single run with starting boundary condition
    g = sysparams['g']
    r = sysparams['r']
    m = sysparams['m']
    k = const['k']
    Cp = sysparams['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
#..........initial values calculations for the run..............................
    xstartrad = mt.radians(xstart)
    V = y0[0]
    T = y0[1]
    Q = y0[2]
    E = y0[3]

    if runparams['RT_factor'] == 0:
        Tf = surf_temp_calc_noRT(xstartrad,sysparams)
    else:
        Tf = surf_temp_calc(xstartrad,T,P,E,sysparams,runparams)
    Pv = total_vap_press_find(runparams['names'],Tf,sysparams,runparams)
    M = V*((1-Beta)/(Beta*Cp*T))**(1/2)

    dthetadeg = runparams['dtheta']
    angle = xstart + dthetadeg
    thetadeg = angle
    dthetarad = mt.radians(dthetadeg)

    names = runparams['names']
    Pv_coefs = sysparams['Pv_coefs']
    
    varnames = runparams['varnames']
    local_vars = {'angle': angle, 'V': V, 'P': P, 'T': T, 'Pv': Pv,'M': M, 'E': E, 'Q': Q, 'Tf': Tf}
    outputs = {}
    for i in np.arange(0,len(varnames)):
        outputs[varnames[i]] = [local_vars[varnames[i]]]
    
    Vnew = V
    Pnew = P
    Tnew = T
    criticalangle = 0
    while thetadeg<=180:
        Vcurrent = Vnew
        Pcurrent = Pnew
        Tcurrent = Tnew
        theta = mt.radians(thetadeg)
        Pv = total_vap_press_find(runparams['names'],Tf,sysparams,runparams)
        fmass,fmom,fenergy = y_calc([Vcurrent,Pcurrent,Tcurrent],theta,sysparams)
        dfmass,dfmom,dfenergy,Q,Tf,E = dy_calc([Vcurrent,Pcurrent,Tcurrent,Q,E],theta,sysparams,runparams)

        fmass2 = ((1/2)*dfmass*dthetarad+fmass)/mt.sin(theta+dthetarad/2)
        fmom2 = ((1/2)*dfmom*dthetarad+fmom)/mt.sin(theta+dthetarad/2)
        fene2 = ((1/2)*dfenergy*dthetarad+fenergy)/mt.sin(theta+dthetarad/2)
        V,P,T,Q,status = VPTsolve(fmass2,fmom2,fene2,Q,E,theta+dthetarad/2,sysparams,runparams)
        
        if status != 1:
            break
        if runparams['flow'] == 'transonic' and np.isnan(Pnew):
            break

        #compute ftemp1 = f(xtemp1,tn+1/2*dt)
        dfmass2,dfmom2,dfene2,Q2,Tf2,E2 = dy_calc([V,P,T,Q,E],theta+dthetarad/2,sysparams,runparams)

        #find xn = xn + dt*dftemp1
        f1 = (dfmass2*dthetarad+fmass)/mt.sin(theta+dthetarad)
        f2 = (dfmom2*dthetarad+fmom)/mt.sin(theta+dthetarad)
        f3 = (dfene2*dthetarad+fenergy)/mt.sin(theta+dthetarad)
        Vnew,Pnew,Tnew,Q,status = VPTsolve(f1,f2,f3,Q2,E2,theta+dthetarad,sysparams,runparams)
        if status != 1:
            break

        M = Vnew*((1-Beta)/(Beta*Cp*Tnew))**(1/2)
        #if M < outputs['M'][-1] and runparams['flow'] == 'subsonic':
         #    status = 2
          #   break
        if runparams['RT_factor'] == 0:
            Tf = surf_temp_calc_noRT(theta+dthetarad,sysparams)
        else:
            Tf = surf_temp_calc(theta+dthetarad,Tnew,Pnew,E,sysparams,runparams)
        Pv = total_vap_press_find(runparams['names'],Tf,sysparams,runparams)
        sysparams,mass_frac = update_comp(Tf,sysparams,runparams)

        thetadeg = thetadeg + dthetadeg

        local_vars = {
            'angle': thetadeg, 'V': Vnew, 'P': Pnew, 'T': Tnew, 'Pv': Pv,
            'M': M, 'E': E, 'Q': Q, 'Tf': Tf
        }
        for i in np.arange(0,len(varnames)):
            outputs[varnames[i]].append(local_vars[varnames[i]])


        

    outputs['status'] = status    
    return outputs


















