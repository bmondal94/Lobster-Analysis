#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:09:09 2020

@author: bmondal
"""
import os
import os.path
import numpy as np      
from scipy import linalg as LA
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.collections

dirname = '/home/bmondal/MyFolder/LobsterAnalysis/GaN' #'GaAs_SP'

NBANDS = NBASISFUNCTION = 8
realcols = np.arange(1,NBANDS+1,dtype=int)
NsymmetryKPTS = 5 # No. of high symmetry KPOINTS (default=5). e.g 5 for L-G-X-U K-G. U and K together considered to be one.
NPOINTS = 40 #No. of intermediate points in each symmetry line section
TotalKP = (NsymmetryKPTS - 1)*NPOINTS # Total number of k-points
nn = (1491-TotalKP)*(NBASISFUNCTION+3)*2

#%%  COHP matrix
filename = dirname+'/KspaceCOHPBand1.lobster'

COHP = np.loadtxt(filename, comments=['#','C','b'], usecols=realcols) #*(-1)
COHPMatrix = np.asarray(np.split(COHP, TotalKP))

#%%  no orthonormalization coefficient matrix
filename = dirname+'/coefficientMatrices.lobster'
Coeff = np.loadtxt(filename, comments=['R','I','c','b'], usecols=realcols,skiprows=nn) 
CoeffMatrix = np.asarray(np.split(Coeff, TotalKP))

# %% Creating the re-arranged hamiltonianMatrices file
wfile = dirname+'/coefficientMatricesLSO2_New.lobster'
if (os.path.isfile(wfile)):
    print('**** File already exists')
    pass
else:
    filename = dirname+'/coefficientMatricesLSO2.lobster'
    dd = {}
    with open(filename) as f:
        lines=f.readlines()
        for ln,line in enumerate(lines):
            if line.startswith('coef'):
                kp = line.split(' ')[6]
                if int(kp)>1331:
                    if kp in dd.keys():
                        dd[kp+'Im']=lines[ln:ln+11]
                    else:
                        dd[kp]=lines[ln:ln+11]
    
    wf=open(wfile,'ab')
    for i in range(1332,1492):
        np.savetxt(wf,dd[str(i)],fmt='%s',newline='')
        np.savetxt(wf,dd[str(i)+'Im'],fmt='%s',newline='')
    
    wf.close()

#%%  Final rthonormalization coefficient matrix
filename = dirname+'/coefficientMatricesLSO2_New.lobster'
Coeff = np.loadtxt(filename, comments=['R','I','c','b'], usecols=realcols) 
CoeffMatrix = np.asarray(np.split(Coeff, TotalKP))

# %% Creating the re-arranged hamiltonianMatrices file
wfile = dirname+'/hamiltonMatrices_New.lobster'
if (os.path.isfile(wfile)):
    print('**** File already exists')
    pass
else:
    filename = dirname+'/hamiltonMatrices.lobster'
    dd = {}
    with open(filename) as f:
        lines=f.readlines()
        for ln,line in enumerate(lines):
            if line.startswith('Ham'):
                kp = line.split(' ')[6]
                if int(kp)>1331:
                    if kp in dd.keys():
                        dd[kp+'Im']=lines[ln:ln+11]
                    else:
                        dd[kp]=lines[ln:ln+11]
    
    
    wf=open(wfile,'ab')
    for i in range(1332,1492):
        np.savetxt(wf,dd[str(i)],fmt='%s',newline='')
        np.savetxt(wf,dd[str(i)+'Im'],fmt='%s',newline='')
    
    wf.close()

#%%  Hamiltonian matrix
filename = dirname+'/hamiltonMatrices_New.lobster'

Ham = np.loadtxt(filename, comments=['R','I','H','b'], usecols=realcols)
HamMatrix = np.asarray(np.split(Ham, TotalKP))

#%% Calculate COHP from coefficient and Hamiltonian matrix

band_index = 1 # band index starts with 1
kpoint = 80 # K-point index starts with 1
H = HamMatrix[kpoint-1,:NBASISFUNCTION] + HamMatrix[kpoint-1,NBASISFUNCTION:]*1j
CC = CoeffMatrix[kpoint-1,:NBASISFUNCTION] + CoeffMatrix[kpoint-1,NBASISFUNCTION:]*1j

C = CC[:,band_index-1]
cohp_cal = C.conjugate().reshape((len(C),1)) * C * H

print("Total COHP calculated = ", cohp_cal.sum())
print("Total COHP from Matrix = ",COHPMatrix[kpoint-1].sum())

#%%
########## Plot the Hamiltonian matrix itself ##############################
orbitals = ["$s_a$","$s_c$","$pz_a$","$pz_c$","$px_a$","$px_c$","$py_a$","$py_c$"]
## Prepare the final hamiltonian
HH = {}
idx = [4,0,6,2,7,3,5,1]
for I in range(40,80):
    HamMatrix_ = HamMatrix[I][:,idx]
    HH[(I-40,0)] = HamMatrix_[:NBASISFUNCTION][[idx]] # Real parts
    HH[(I-40,1)] = HamMatrix_[NBASISFUNCTION:][[idx]] # Imaginary parts

t=list(HH.values())
vminn, vmaxx = np.amin(t), np.amax(t)

#%%
xy = [[2,0],[3,1],[0,2],[1,3],[6,4],[7,5],[4,6],[5,7]]
patches = [plt.Circle(center, 0.5) for center in xy]
ticks = np.arange(NBASISFUNCTION)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6)) 
title_text = "Total Hamiltonian matrix\n Real part \t\t\t\t\t\t\t\t\t  Imaginary part ".expandtabs()
fig.suptitle(title_text)                              

for j in range(2):
    axs[j].set_xticks(ticks)
    axs[j].set_yticks(ticks)
    axs[j].set_xticklabels(orbitals,fontsize=10)
    axs[j].set_yticklabels(orbitals,fontsize=10)
    plt.setp(axs[j].get_xticklabels(), rotation=90, ha="center")
    axs[j].vlines([1.5,3.5,5.5],[-0.5,-0.5,3.5],[3.5,7.5,7.5],colors='k')
    axs[j].hlines([1.5,3.5,5.5],[-0.5,-0.5,3.5],[3.5,7.5,7.5],colors='k')
#fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])
coll = matplotlib.collections.PatchCollection(patches, facecolors='black')
axs[0].add_collection(coll)
coll = matplotlib.collections.PatchCollection(patches, facecolors='black')
axs[1].add_collection(coll)
axs[1].axline((0,0),(7,7), c='k',lw=3)

im = []; texts = []
def init():
    for j in range(2):
        im.append(axs[j].matshow(HH[(0,j)], vmin=vminn, vmax = vmaxx))
        # Loop over data dimensions and create text annotations.
        for k in ticks:
                for l in ticks:
                    texts.append(axs[j].text(l, k, "{:.2f}".format(HH[(0,j)][k, l]),ha="center", va="center", color="w"))
        texts.append(axs[j].text(k/2,k+1,"k-point {kp}".format(kp=0),fontsize=14,ha="center", va="center",))
        
    ccbar=fig.colorbar(im[0], cax=cbar_ax)
    ccbar.ax.tick_params(labelsize=10)
    return im, texts

def updateData(I):
    ii=0
    for j in range(2):
        im[j].set_data(HH[(I,j)])
        for k in ticks:
            for l in ticks:
                texts[ii].set_text("{:.2f}".format(HH[(I,j)][k, l])) #,ha="center", va="center", color="w")
                ii+=1
        texts[ii].set_text("k-point {kp}".format(kp=I))
        ii+=1
    return im, texts       
    

def onClick(event):
    global anim_running
    if anim_running:
        simulation.event_source.stop()
        anim_running = False
    else:
        simulation.event_source.start()
        anim_running = True
        
anim_running = True   
fig.canvas.mpl_connect('key_press_event', onClick)
fig.canvas.mpl_connect('button_press_event', onClick)
simulation = animation.FuncAnimation(fig, updateData,init_func=init, frames=40, interval=500, repeat=False, blit=False) #, repeat_delay=1000)


savefig = 1
if savefig:
    print('*** Saving movie')
    movdirname = "/home/bmondal/MyFolder/LobsterAnalysis/Movies/GaN"
    filname = movdirname+'/GaN_SP_H-matrix.mp4'
    simulation.save(filname, fps=2, dpi=300)

#%% Plot the different contributions individually
print(orbitals)
def myfig():
    fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(12,6))
    ax[0].set_xlim(0,39)
    ax[0].set_ylabel('E (eV)')
    ax[0].set_xlabel(r"K-points ($\Gamma \rightarrow X$)")
    ax[1].set_xlabel(r"K-points ($\Gamma \rightarrow X$)")
    ax[2].set_xlabel(r"K-points ($\Gamma \rightarrow X$)")
    ax[0].set_xticks([])
    ax[0].set_title("Real part")
    ax[1].set_title("Imaginary part")
    ax[2].set_title("Magnitude (Re+Im)")
    return fig,ax

#%% Contributions from s-p 4x4 matrix
fig,ax = myfig()
fig.suptitle('H-matrix elemets (upper triangle part) [s-p 4x4]')
for I in range(0,4): # (0,4)==upto s-p; use range(0,2) for only s-s part and (2,4) for pz-pz part
    J=0
    while J <= I :
        x = np.array([[HH[(j,0)][I,J],HH[(j,1)][I,J]] for j in range(40)])
        ax[0].plot(x[:,0], label=orbitals[I]+'-'+orbitals[J])
        ax[1].plot(x[:,1])
        #ax[2].plot(x[:,0]+x[:,1]) # Straight forward addition
        ax[2].plot(abs(x[:,0]+x[:,1]*1j)) # Absolute value in magnitude
        J+=1

ax[0].legend(ncol=2)

#%% Contributions from p-p 4x4 matrix

fig,ax = myfig()
fig.suptitle('H-matrix elemets (upper triangle part) [p-p 4x4]')
for I in range(4,8):
    J=4
    while J <= I :
        x = np.array([[HH[(j,0)][I,J],HH[(j,1)][I,J]] for j in range(40)])
        ax[0].plot(x[:,0], label=orbitals[I]+'-'+orbitals[J])
        ax[1].plot(x[:,1])
        ax[2].plot(x[:,0]+x[:,1]) # Straight forward addition
        #ax[2].plot(abs(x[:,0]+x[:,1]*1j)) # Absolute value in magnitude
        J+=1

ax[0].legend(ncol=2)

