#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time,sys,traceback,math

LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE="/home/public/youran_FQHE/Fqhe.log"
def log(msg,l=1,end="\n",logfile=LOGFILE):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s.%03d"%(time.strftime("%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    perfix="%s [%s,%s:%03d]"%(now_str,lstr,st.name,st.lineno)
    if l<3:
        tempstr="%s %s%s"%(perfix,str(msg),end)
    else:
        tempstr="%s %s:\n%s%s"%(perfix,str(msg),traceback.format_exc(limit=5),end)
    print(tempstr,end="")
    if l>=1:
        with open(logfile,"a") as f:
            f.write(tempstr)

import numpy,itertools

# Number of electrons
Ne=32
# Laughlin m
m_lauphlin=3
# Filling fraction
nu=1/3
# mass for CF, under the nature unit me=1
Mcf=0.067
# Sample Radius
Rsmp=math.sqrt(Ne/(nu*math.pi))
log("Sample radius when Ne=%d, \\nu=%.4f: %.4f"%(Ne,nu,Rsmp))
# magnetic length under this unit
lb=1/math.sqrt(2*math.pi)
# Compute boundry size
# for 32 electrons, 99.9% of them are in Rsmp+0.3*lb, 99.99% are in Rsmp+0.5*lb
Lcut=2*(Rsmp+1*lb)
#Lcut=math.sqrt(Ne/nu)
# Discrete number and step size in each direction
Npts=200
Lstep=Lcut/Npts
log("m, nu, Mcf: %d, %.3f, %.3f"%(m_lauphlin,nu,Mcf))
log("Cut Length, discrete step size: %.4f, %.4f"%(Lcut,Lstep))
# weight for Ay
ax_wt=0.5
ay_wt=0.5
log("ax_wt, ay_wt: %.1f, %.1f"%(ax_wt,ay_wt))
# vacuum permittivity
# using the data for GaAs
ep0=2.585105102e-3*12.6
# background B
B0=1

def calc_nposi(i,j,R):
    """
        for computing the positive disk density
    """
    x=j-(Npts-1)/2;y=i-(Npts-1)/2
    Rxy=math.sqrt(x**2+y**2)
    dR=R/Lstep-Rxy
    if dR>math.sqrt(1/2):
        return 1.0
    elif dR<-math.sqrt(1/2):
        return 0.0
    else:
        x=abs(x);y=abs(y)
        x,y=(y,x) if y>x else (x,y)
        ct=x/Rxy;st=y/Rxy
        assert ct>=st>=0
        dR0=(ct-st)/2;dR1=(ct+st)/2
        if dR<-dR1:
            return 0.0
        elif dR<-dR0:
            return (1/2)*(st/ct)*(1-(abs(dR)-dR0)/st)**2
        elif dR<dR0:
            return (1/2)*(st/ct+2*(dR+dR0)/ct)
        elif dR<dR1:
            return 1-(1/2)*(st/ct)*(1-(abs(dR)-dR0)/st)**2
        else:
            return 1.0

# charge density for positive disk
n_posi=numpy.zeros((Npts,Npts))
for i,j in itertools.product(range(Npts),range(Npts)):
    n_posi[i,j]=calc_nposi(i,j,Rsmp)
#print((n_posi.sum()-math.pi*(Rsmp/Lstep)**2))
n_posi*=-Ne/(n_posi.sum()*Lstep**2)

# for integrating charges of interior
Rinter=5*lb
mask_inter=numpy.zeros((Npts,Npts))
for i,j in itertools.product(range(Npts),range(Npts)):
    mask_inter[i,j]=calc_nposi(i,j,Rinter)

def calc_ewt(i,j):
    """
        for computing the columb potentital from density
    """
    wtdict={(0,0):3.7393513,
            (1,0):1.0952306,(1,1):0.7236673,
            (2,0):0.50709472,(2,1):0.45178021,(2,2):0.35572797,
            (3,0):0.33529131,(3,1):0.3178536,(3,2):0.27841416,(3,3):0.2363524
            # no need for higher order
            #(4,0):0.25080609,(4,1):0.24326438,(4,2):0.22416866,(4,3):0.20039878,(4,4):0.17705182
            }
    i=abs(i);j=abs(j)
    if (i,j) in wtdict:
        return wtdict[(i,j)]
    elif (j,i) in wtdict:
        return wtdict[(j,i)]
    else:
        assert i>3 or j>3
        r=math.sqrt(i**2+j**2)
        return 1/r+1/(20*r**3)# no need for higher order +(-4*i**2+27*i**2*j**2-4*j**2)/(280*r**9)

# for computing Columb potential
# ewt_tab = electric weight table
ewt_tab=numpy.zeros((2*Npts-1,2*Npts-1))
for i,j in itertools.product(range(2*Npts-1),range(2*Npts-1)):
    ewt_tab[i,j]=calc_ewt(Npts-1-i,Npts-1-j)