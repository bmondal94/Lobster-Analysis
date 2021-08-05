#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:12:28 2020

@author: bmondal
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

params = {'legend.fontsize': 18,
          'figure.figsize': (10,8),
         'axes.labelsize': 18,
         'axes.titlesize': 18,
         'xtick.labelsize': 18,
         'ytick.labelsize': 18,
         'errorbar.capsize':2}
plt.rcParams.update(params)
np.set_printoptions(precision=3,suppress=True)
#%%

#mdir = '/home/bmondal/MyFolder/LobsterAnalysis/GaN/'
#mdir = '/home/bmondal/MyFolder/LobsterAnalysis/GaAs/'
mdir = '/home/bmondal/mob43741/LOBSTER/Projects/GaP/'
nkpoints = 160
NsymmetryKPTS = 5
minval = -13 ; maxval = 11 # Defines the energy rage of bands in plot
CBindex = 10 # Because band array start with 0 index
NPOINTS = int(nkpoints / (NsymmetryKPTS-1))
filekp = mdir+'KPOINTS'
kp1 = np.genfromtxt(filekp, skip_header=3)
kp = kp1[-nkpoints:,:-1]

filepos = mdir+'POSCAR'
def ReadPoscar(poscar):
    filename=open(poscar,"r")
    lines=filename.readlines()
    filename.close()
    factor=float(lines[1].split()[0])
    a=np.asarray(lines[2].split()).astype(np.float)
    b=np.asarray(lines[3].split()).astype(np.float)
    c=np.asarray(lines[4].split()).astype(np.float)
    vector=np.array([a,b,c])
    vector=vector*factor
    ion = np.array(lines[5].split(), dtype=str)
    ionnumber = np.array(lines[6].split(), dtype=int)
    return vector, ion, ionnumber

B,iontype,iontypenumber = ReadPoscar(filepos)
reciprocal_lattice = np.linalg.inv(B).T
print("Reciprocal Lattice vectors: \n",reciprocal_lattice)

KP = np.dot(kp,reciprocal_lattice)
vkpts = np.split(KP,NsymmetryKPTS-1)
x = np.cumsum(np.insert(np.linalg.norm(np.diff(vkpts,axis=1),axis=2),0,0,axis=1))
SpecialKPOINTS = np.append(x[::NPOINTS],x[-1])

SpecialKPTS = ["L","$\Gamma$","X","U,K","$\Gamma$"]


#%%############## FATBAND part ##################

onlyfatband = 1
bandplot = 0
bandorbital = False  # band + orbital contribution

#%%########### Plotting #####################
def definefigure():
    fig, ax = plt.subplots()
    ax.set_xticks(SpecialKPOINTS)
    ax.set_xticklabels(SpecialKPTS)
    ax.set_xlabel('k-point')
    ax.set_xlim(left=0, right=SpecialKPOINTS[-1])
    ax.tick_params(axis='y',which='both')
    return fig,ax

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
#%%

if onlyfatband:
    drawnewfig = True
    #bandlabel = ''
    while True:
        if drawnewfig:
            fig,ax = definefigure()
            ax.vlines(SpecialKPOINTS,0,100,linestyles='--',color='k')
            ax.set_ylim(bottom=-2, top=102)
        while True:
            orbitalname = input("Which orbital (e.g. 1s, 4p or s, p [note: For separate atom analysis l-quantum no. is must]) do you want to analyze?   ")
            seperateatom = input("Do you want to analze for separate atom? If not press ENTER.  ")
            if seperateatom:
                atomname = input("Which atom (with index) do you want to analyze?   ")
                orbitalname = atomname+'_'+orbitalname
                filen=mdir+'FATBAND_'+orbitalname+'.lobster'
                print(filen)
                if not (os.path.isfile(filen)):
                    sys.stderr.write('\nError: No FATBAND file exist with this orbital.')
                    sys.exit()
                else:
                    #bandlabel += ' '+atomname
                    data = np.genfromtxt(filen,usecols=2)
            else:
                #bandlabel += ' Total'
                FATbandFilelist = glob.glob(mdir+"FATBAND_*")
                FATbandFile = [fi for fi in FATbandFilelist if (orbitalname+'.lobster') in os.path.basename(fi)]
                #print(FATbandFile)
                if (len(FATbandFile) == 0 ):
                    sys.stderr.write('\nError: No FATBAND file exist with this orbital.')
                    sys.exit()
                else:
                    data = np.genfromtxt(FATbandFile[0],usecols=2)
                    if (len(FATbandFile) > 1 ):
                        for filen in FATbandFile[1:]:
                            data += np.genfromtxt(filen,usecols=2) 
    
            fatband = np.array(np.split(data,nkpoints))
            NBANDS = len(fatband[0])
    
            #ax.set_title(bandlabel)
            ax.set_ylabel('Orbital contribution (%)')
            print("\n*** Which band index you want to plot? Band index starts with 1.")
            print("*** If you are finished with providing band index then press ENTER to continue next.")
            while True:
                bandindex = input("Band Index: ")
                if bandindex:
                    y = fatband[:,int(bandindex)-1] * 100.0
                    #minval = min(y); maxval = max(y)
                    ax.set_ylim(bottom=0, top=100)
                    ax.plot(x,y,'-',label='band-'+bandindex+'_'+orbitalname)
                    #ax.plot(x,y,'--',label=orbitalname[0:2]+'-'+orbitalname[4:])
                    ax.legend(ncol=1)
                else:
                    break
            nextround = input("Do you want to go for next orbital analysis? If not press ENTER.  ")
            if not nextround:
                break
            
        furtherdraw = input("Do you want to plot next file? If not press ENTER.  ")
        if furtherdraw:
            drawnewfig = input("Do you want new plot? If not press ENTER.  ")
        else:
            break

        
if bandplot:
    print("I am in band loop")
    FATbandFile = glob.glob(mdir+"FATBAND_*")
    datae = np.genfromtxt(FATbandFile[0],usecols=1) 
    energy_scale = np.array(np.split(datae,nkpoints)) 
    NBANDS = len(energy_scale[0])

    
    fig,ax =definefigure()
    ax.set_ylabel('Energy (eV)')
    ax.set_ylim(bottom=minval, top=maxval)
    #ax.set_xlim(SpecialKPOINTS[1],SpecialKPOINTS[2]) #G-X
    ax.axhline(color='k',ls='--')
    ax.vlines(SpecialKPOINTS,minval,maxval,linestyles='--',color='k')
    
    if not bandorbital:
        differentiatebandline = False
        if (differentiatebandline):
            linestyles = ['-', '--', '-.', ':',(0, (5, 10)),(0, (3, 10, 1, 10)),(0, (3, 1, 1, 1)),(0, (3, 10, 1, 10, 1, 10))]
            for lnd, linestyle in enumerate(linestyles):
                ax.plot(x,energy_scale[:,lnd],linestyle=linestyle)
        else:
            ax.plot(x,energy_scale,'k-')
            ax.plot(x,energy_scale[:,CBindex-1],'-') # Valence band
            ax.plot(x,energy_scale[:,CBindex],'-') # Conduction band 
            #ax.plot(x,energy_scale[:,CBindex-1],'m-') # Valence band
            #ax.plot(x,energy_scale[:,CBindex+3],'m-') # Conduction band +3         
    else:
        dataframe = {}
        for finame in FATbandFile:
            key = os.path.basename(finame)
            dataframe[key] = np.array(np.split(np.genfromtxt(finame,usecols=2), nkpoints))
        seperatecontrib = input("Do you want separate atom contributions or not? If not press ENTER.  ")
        if seperatecontrib:
            while True:
                atomname = input("Which atom (with index) do you want to analyze?   ")
                if atomname:
                    orbitalname = input("Which orbital do you want to analyze?   ")
                    y = dataframe['FATBAND_'+atomname+'_'+orbitalname+'.lobster']
                    for J in range(NBANDS):
                        colorline(x, energy_scale[:,J],y[:,J])
                    #ax.legend(label=atomname+'_'+orbitalname)
                else:
                    break   
        else:
            sys.exit()
            #for J in len(energy_scale):
            #       colorline(x, energy_scale[:,J],y[:,J])


plt.show()

#%%%%%%%%%%%%%%%%%%%
# Simultaneous optimization using iteration from Tight binding model

from scipy.optimize import curve_fit

k=np.linspace(0,1,41)

# G-X valebce band and last band
vba = energy_scale[40:81,CBindex-1]
vba2 = energy_scale[40:81,-1]
av = (vba+vba2)*.5

fig1, ax1 = plt.subplots()
ax1.plot(k,vba,'o',label='2vb')
ax1.plot(k,vba2,'o',label='2(cb+2)')
ax1.plot(k,av, 'k--', label='average')
ax1.set_title(r"$\Gamma \to$ X GaAs")
ax1.set_xlabel("Energy (eV)")
ax1.set_ylabel("k-point")

E1 = 1.0414; E2 = 2.16686
E3 =1.9546; E4 = 6.534
alpha = 0.66; beta =0.66

def func1(k, E1,E2,alpha,beta,E3,E4):
    F = E1+alpha*k**2
    K = E2+beta*k**2
    H2 = E3*E3*(np.cos(np.pi*0.5*k)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k)**2)
    f1 = ((F+K)*0.5)+ np.sqrt(((F-K)*0.5)**2 + H2-J2)
    return f1
    
def func2(k, E1,E2,alpha,beta,E3,E4):
    F = E1+alpha*k**2
    K = E2+beta*k**2
    H2 = E3*E3*(np.cos(np.pi*0.5*k)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k)**2)
    f2 = ((F+K)*0.5)- np.sqrt(((F-K)*0.5)**2 + H2-J2)
    return f2


poptt=[E1,E2,alpha,beta,E3,E4]
eps = [1.e-1]*len(poptt)
perr1 = np.zeros(len(poptt))
perr2 = np.zeros(len(poptt))
iteration=0

plt.ion()
fig2, ax2 = plt.subplots()
ax2.set_title("CB")
fig3, ax3 = plt.subplots()
ax3.set_title("VB")
fig4, ax4 = plt.subplots()
ax4.set_title("Difference between 2 set of optimization parameters")
while True:
    y = vba2
    popt1, pcov = curve_fit(func1, k, y, p0=poptt, bounds=(0,[10,10,2,2,10,10]))
    perr1 -= np.sqrt(np.diag(pcov))
    ax1.plot(k, func1(k,*popt1))
    
    y = vba
    popt2, pcov = curve_fit(func2, k, y, p0=popt1, bounds=(0,[10,10,2,2,10,10]))
    perr2 -= np.sqrt(np.diag(pcov))
    ax1.plot(k, func2(k,*popt2))
    
    poptt = popt2
    dif = popt1-popt2
    iteration+=1
    
    ax2.plot(perr1)
    ax3.plot(perr2)
    ax4.plot(dif)
    plt.draw()
    
    if np.all(abs(dif) < eps):
        print("reached optimization")
        break
    elif (iteration > 10):
        print("No optimization")
        print("Difference bw 2 set of optimization parameters:", dif)
        break
    else:
        pass
    
    plt.pause(0.001)
        
ax1.plot(k, func1(k,*popt1), 'k--')
ax1.plot(k, func2(k,*popt1), 'k--')

#%% optimization simultanously by hstack
from scipy.optimize import curve_fit

k=np.linspace(0,1,41)

# G-X valebce band and last band
vba = energy_scale[40:81,CBindex-1]
vba2 = energy_scale[40:81,-1]
av = (vba+vba2)*.5
plt.figure(1)
plt.plot(k,vba,'o',c='r')
plt.plot(k,vba2,'o',c='r',label='DFT')
plt.plot(k,av, 'k--', label='Average (DFT)')

## Curve fit
f = lambda k, shift, alpha: shift + alpha*k*k #K^2 
f = lambda k, shift, alpha: shift + alpha*np.cos(np.pi*k) # a*cos(K)
f = lambda k, shift, alpha, beta: shift + alpha*np.cos(np.pi*k)+ beta**np.sin(np.pi*0.5*k) # a*cos(K)+b*sin(K)
f = lambda k, shift, alpha: shift + np.exp(alpha*k)*np.cos(np.pi*k)
popt, pcov = curve_fit(f, k, av)
plt.plot(k,f(k, *popt), '.-', label='Average (DFT)-curve-fit')
## Polynomial fit
z = np.poly1d(np.polyfit(k, av, 10))
plt.plot(k,z(k), '.-', label='Average (DFT)-poly-fit')

#plt.title(r"$\Gamma \to$X GaAs")
plt.ylabel("Energy (eV)")
plt.xlabel("k-point")
plt.xlim(0,1)
plt.xticks([0,1],["$\Gamma$",'X'])
plt.legend()

E1 = 1.0414; E2 = 2.16686
E3 =1.9546; E4 = 6.534
alpha = 0.66; beta =0.66

def func1(k, E1,E2,alpha,beta,E3,E4):
    #F = E1+alpha*k**2
    #K = E2+beta*k**2
    F = E1+alpha*np.cos(np.pi*k)
    K = E2+beta*np.cos(np.pi*k)
    #F = E1+alpha * z(k)
    #K = E2+beta*z(k)
    
    H2 = E3*E3*(np.cos(np.pi*0.5*k)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k)**2)
    f1 = ((F+K)*0.5)+ np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f2 = ((F+K)*0.5)- np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f = np.hstack([f1,f2])
    return f
    
yd = np.hstack([vba2, vba])
xd = np.hstack([k,k])
popt, pcov = curve_fit(func1, k, yd, p0=[E1,E2,alpha,beta,E3,E4], bounds=(0,[10,10,np.inf,np.inf,10,10]))

adata = func1(k,*popt)
plt.plot(k,adata[:len(k)],'m')
plt.plot(k,adata[len(k):],'m',label='sp3 quadratic eqn. fit')
plt.legend()
print("Optimized parameters [p1, p2, alpha, beta, vxx, vxy]:\n", popt)


#%%
from math import pi
import numpy as np

k1=np.linspace(0,1,41)

band1 = energy_scale[40:81,10]
band2 = energy_scale[40:81,11]
band3 = energy_scale[40:81,14]
band4 = energy_scale[40:81,15]

band3_ = np.hstack([band3[:12],band4[12:]])
band4_ = np.hstack([band4[:12],band3[12:]])

band = np.hstack([band1, band2, band3, band4])
k1_ = np.hstack([k1, k1, k1, k1])
y = np.zeros(len(k1_))

a=-10.3431; c=-0.09569; f=1.0414; k=2.16686
vss=-5.0513; vxx=1.9546
vsaspga=1.48; vsgapas=1.839 
initial_guess = [a,c,f,k,vxx,vss,vsgapas,vsaspga]

def f4by4(xx,a,c,f,k,vxx,vss,vsgapas,vsaspga):
    x,k1 = xx
    k11 = pi*0.5*k1
    ck1 = np.cos(k11)**2
    sk1 = np.sin(k11)**2
    ff=(a-x)*(c-x)*(f-x)*(k-x) - (a-x)*(c-x)*vxx*vxx*ck1 \
                            - (f-x)*(k-x)*vss*vss*ck1 \
                            - (a-x)*(k-x)*vsgapas*vsgapas*sk1 \
                            - (f-x)*(c-x)*vsaspga*vsaspga*sk1 \
        +(vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2
        
    return ff

poptt,pconvv = curve_fit(f4by4, (band1,k1), np.zeros(len(k1)), p0=initial_guess)
#poptt,pconvv = curve_fit(f4by4, (band,k1_), y, p0=initial_guess)
f4by4((band1,k1),*poptt)

a,c,f,k,v_xx,v_ss,v_sGa_pAs,v_sAs_pGa = poptt


plt.figure(1)
plt.plot(k1,band1,'o')
plt.plot(k1,band2,'o')
plt.plot(k1,band3_,'o')
plt.plot(k1,band4_,'o')

#===============================================================================
#%% Quartic equation solution
def f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,vsaspga, KEe1, KEe2, KEe3, KEe4, n_band):
    k11 = pi*0.5*k1
    k1_2 = z(k1)
    k1_2 = np.cos(k11*2) 
    #k1_2 = k1*k1
    ck1 = np.cos(k11)**2
    sk1 = np.sin(k11)**2
    
    KE1, KE2, KE3, KE4 = KEe1*k1_2, KEe2*k1_2, KEe3*k1_2, KEe4*k1_2    
    a, c, f, k = a+KE1, c+KE2, f+KE3, k+KE4
    
    vxxck1 = vxx*vxx*ck1
    vssck1 = vss*vss*ck1
    vsgapassk1 = vsgapas*vsgapas*sk1
    vsaspgask1 = vsaspga*vsaspga*sk1
    
    B = -(a+c+f+k)
    C = (a*c+f*k)+(a+c)*(f+k)-(vxxck1+vssck1+vsgapassk1+vsaspgask1)
    D = -(a*c*(f+k)+f*k*(a+c)) + (a+c)*vxxck1 + (f+k)*vssck1 + (a+k)*vsgapassk1\
        + (f+c)*vsaspgask1
    E = a*c*f*k - a*c*vxxck1 - f*k*vssck1 - a*k*vsgapassk1 - f*c*vsaspgask1 + \
        (vss*vxx*ck1 - vsgapas*vsaspga*sk1)**2
    
    p = (8*C - 3*B*B)/8.
    q = (B*B*B - 4*B*C + 8*D)/8.
    Dt = 256*E*E*E - 192*B*D*E*E - 128*C*C*E*E + 144*C*D*D*E - 27*D*D*D*D +\
        144*B*B*C*E*E -6*B*B*D*D*E - 80*B*C*C*D*E + 18*B*C*D*D*D +16*C*C*C*C*E+\
        -4*C*C*C*D*D -27*B*B*B*B*E*E +18*B*B*B*C*D*E -4*(B*D)**3 -4*B*B*C*C*C*E +B*B*C*C*D*D
    Dt_0 = C*C - 3*B*D + 12*E
    Dt_1 = 2*C*C*C - 9*B*C*D + 27*B*B*E + 27*D*D - 72*C*E
    
    if (Dt.any() < 0):
       print(r"$\Delta<0$ case is not allowed. This will give always 2 complex roots")
       sys.exit()
    elif(Dt.all() >0):
        phi = np.arccos(0.5*Dt_1/(Dt_0**(1.5)))
        S =0.5*np.sqrt(-(2/3.)*p + (2/3.)*np.sqrt(Dt_0)*np.cos(phi/3.))
    else:
        print(r"$\Delta=0$ multiple root")
        sys.exit()
    
    root_dic = {}
    root_dic[0] = -(B/4.) -S - 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[1] = -(B/4.) -S + 0.5*(np.sqrt(-4*S*S-2*p+(q/S)))
    root_dic[2] = -(B/4.) +S - 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    root_dic[3] = -(B/4.) +S + 0.5*(np.sqrt(-4*S*S-2*p-(q/S)))
    
    root = []
    for I in n_band: # THis array should be same as 'no_of_band' below
        root = np.hstack([root, root_dic[I]])

    return root

def func_p(k1, E4, E1,E2,alpha,beta,E3):
    F = E1+alpha*k1**2
    K = E2+beta*k1**2
    H2 = E3*E3*(np.cos(np.pi*0.5*k1)**2)
    J2 = -E4*E4*(np.sin(np.pi*0.5*k1)**2)
    f1 = ((F+K)*0.5)+ np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f2 = ((F+K)*0.5)- np.sqrt(((F-K)*0.5)**2 + H2-J2)
    f = np.hstack([f1,f2])
    return f
    

#%%
no_of_band =[0,1,2,3] 
band_d = {}
band_d[0] = energy_scale[40:81,10]
band_d[1] = energy_scale[40:81,11]
band_d[2] = energy_scale[40:81,14]
band_d[3] = energy_scale[40:81,15]
#band_d[2] = np.hstack([band3[:12],band4[12:]])
#band_d[3] = np.hstack([band4[:12],band3[12:]])
p_band = np.hstack([energy_scale[40:81,12], energy_scale[40:81,17]])

a=-10.3431; c=-0.09569; f=1.0414; k=2.16686
vss=-5.0513; vxx=1.9546; vxy = 6.534
vsaspga=1.48; vsgapas=1.839 
initial_guess = [a,c,f,k,vxx,vss,vsgapas,vsaspga]
kin=[0,0,0,0]
bound_ig=([-12,-5,0,0,0,-10,0,0],[1,5,5,6,10,1,10,10])
bound_all=([-12,-5,0,0,0,-10,0,0,-2,-2,-2,-2],[1,5,5,6,10,1,10,10,2,2,2,2])

band, k1_ = [], []
for I in no_of_band:
    band = np.hstack([band,band_d[I]])
    k1_ = np.hstack([k1_, k1])

# First fit the 4x4 matrix bands then use the optimized parameter from there to
# optimize 2x2 matrix bands
fit_ke = 1
if fit_ke:
    poptt,pconvv = curve_fit(lambda k1,a,c,f,k,vxx,vss,vsgapas,vsaspga,KE1,\
                             KE2,KE3,KE4: f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,\
                            vsaspga,KE1,KE2,KE3,KE4,no_of_band), k1, band, \
                            p0=initial_guess+kin, bounds=bound_all)
    p_pass_param = [poptt[2],poptt[3],poptt[-2],poptt[-1],poptt[4]]
    final_paramt = np.copy(poptt)
else:
    poptt,pconvv = curve_fit(lambda k1,a,c,f,k,vxx,vss,vsgapas,vsaspga: \
                             f4by4root(k1,a,c,f,k,vxx,vss,vsgapas,vsaspga, \
                            kin[0], kin[1], kin[2], kin[3], no_of_band), \
                            k1, band, p0=initial_guess, bounds=bound_ig)
    p_pass_param = [poptt[2],poptt[3],kin[2],kin[3],poptt[4]]
    final_paramt = np.hstack([poptt,kin])
popt, pcov = curve_fit(lambda k1, E4: func_p(k1,E4,*p_pass_param), k1, p_band, p0=vxy, bounds=(0,10))

print("Best fit parameters:\n$s_{As}$=%6.3f, s_{Ga}=%6.3f, p_{As}=%6.3f, p_{Ga}=%6.3f\n\
V_{xx}=%6.3f, V_{ss}=%6.3f, V_{sGa-pAs}=%6.3f, v_{sAs-pGa}=%6.3f\n\
KE_{sAs}=%6.3f, KE_{sGA}=%6.3f, KE_{pAs}=%6.3f, KE_{pGa}=%6.3f\n\
V_{xy}==%6.3f"%tuple(np.hstack([final_paramt,popt])))


#%%
if fit_ke:
    plot_y = np.split(f4by4root(k1,*poptt,[0,1,2,3]), 4)
else:
    plot_y = np.split(f4by4root(k1,*poptt,kin[0],kin[1],kin[2],kin[3],[0,1,2,3]), 4)

# For plotting do you want to plot the re-fitted p-double degenerate bands 
# Or no re-fitting?
optimized_p = 1
if optimized_p:
    pdata = func_p(k1,*popt,*p_pass_param)
else:
    pdata = func_p(k1,vxy,*p_pass_param)
    
plt.figure()
plt.ylabel("Energy (eV)")
plt.xlabel("k-point")
plt.xlim(0,1)
plt.xticks([0,1],["$\Gamma$",'X'])
for I in range(4):
    plt.plot(k1,band_d[I],'o-')
    plt.plot(k1,plot_y[I],'.-')
    
# # 2 fold degenerate p-bands
# for I in range(2):
#     #plt.plot(k1,p_band[I*len(k1):(I+1)*len(k1)],'o-')
#     plt.plot(k1,pdata[I*len(k1):(I+1)*len(k1)],'.--',color='gray')


#plt.legend()
#===============================================================================
