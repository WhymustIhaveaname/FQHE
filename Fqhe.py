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

import math,numpy,scipy,scipy.linalg,scipy.sparse.linalg
import itertools,pickle,os
from scipy import sparse
import matplotlib.pyplot as plt

# Number of electrons
Ne=10
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
Npts=128
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
# R
Rinter=5*lb

# positive disk
def calc_nposi(i,j,R):
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

n_posi=numpy.zeros((Npts,Npts))
for i,j in itertools.product(range(Npts),range(Npts)):
    n_posi[i,j]=calc_nposi(i,j,Rsmp)
#print((n_posi.sum()-math.pi*(Rsmp/Lstep)**2))
n_posi*=-Ne/(n_posi.sum()*Lstep**2)

mask_inter=numpy.zeros((Npts,Npts))
for i,j in itertools.product(range(Npts),range(Npts)):
    mask_inter[i,j]=calc_nposi(i,j,Rinter)

def gen_Vimp(locxy,h,q):
    """
        q: positive q stands for electron
    """
    locx,locy=locxy
    log("generating impurity potential: xy=(%.2f,%.2f) h=%.4f q=%.1f"%(locx,locy,h,q))
    hsq=h*h
    Vimp=numpy.zeros((Npts,Npts))
    for i,j in itertools.product(range(Npts),range(Npts)):
        rsq=((i+0.5)*Lstep-locy)**2+((j+0.5)*Lstep-locx)**2
        Vimp[i,j]=q/math.sqrt(rsq+hsq)
    Vimp/=(4*math.pi*ep0)
    return Vimp

# compute Columb potential
def calc_ewt(i,j):
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

ewt_tab=numpy.zeros((2*Npts-1,2*Npts-1))
for i,j in itertools.product(range(2*Npts-1),range(2*Npts-1)):
    ewt_tab[i,j]=calc_ewt(Npts-1-i,Npts-1-j)

def heatmap(mats,titles=None,savename=None):
    # (6.4,4.8) is the default size of plt
    fig,axs=plt.subplots(1,len(mats),figsize=(6.0*len(mats),4.8))
    plots=[]
    for i in range(len(mats)):
        p=axs[i].matshow(mats[i],cmap='hot')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        fig.colorbar(p,ax=axs[i],shrink=0.8)
        if titles:
            axs[i].set_title(titles[i],fontsize=20)
        plots.append(p)
    plt.show()
    if savename:
        fig.savefig(savename)

def plot_profile(n):
    midpt=Npts/2-0.5
    rs=[];ns=[]
    for i,j in itertools.product(range(Npts),range(Npts)):
        r=math.sqrt((i-midpt)**2+(j-midpt)**2)*Lstep
        if r>Lcut/2+lb:
            continue
        rs.append(r)
        ns.append(n[i,j])
    xticks=[];xlabels=[]
    nlb=0
    while nlb*lb<Lcut/2:
        xticks.append(nlb*lb)
        xlabels.append("%d"%(nlb))
        nlb+=1
    plt.figure(figsize=(12,4))
    plt.plot(rs,ns,'.')
    plt.axvline(x=Rsmp)
    plt.title("Density Profile for $m=%d$, $\\nu=%.4f$"%(m_lauphlin,nu),fontsize=20)
    plt.xlabel('$r/l_B$',fontsize=20)
    plt.ylabel('$\\nu_{local}$', fontsize=20)
    plt.ylim(0,0.6)
    plt.xlim(0,Lcut/2+lb)
    plt.xticks(xticks,xlabels)
    plt.grid()
    plt.show()

class Fqhe():
    def __init__(self,n,eigvec,VT_switch=False,VTrn=1.0,Vimp=None):
        self.n=n
        self.VT_switch=VT_switch
        if Vimp is not None:
            self.Vimp=Vimp
        else:
            self.Vimp=numpy.zeros(self.n.shape)
        log("VT: %s, VT_renormalization: %.1f"%(self.VT_switch,VTrn))

        # some parameters from Hu PRL 123, 176802 (2019)
        # will be used in gen_Vxc
        eu=1/(4*math.pi*ep0*lb)
        self.a=-0.78213*eu;self.b=0.2774*eu;self.f=0.33*eu;self.g=-0.04981*eu
        log("eu, a, b, f: %.4f, %.2f, %.2f, %.2f"%(eu,self.a,self.b,self.f))

        self.update_n(n,1)
        if self.VT_switch:
            self.VTrn=VTrn
            self.VT_last=numpy.zeros(self.n.shape)
            self.VT_reset_flag=True

    def update_n(self,n_neo,lr):
        self.lr=lr
        self.n=n_neo*self.lr+self.n*(1-self.lr)
        self.n_updated=True
        self.A_updated=False
        self.gen_A()

    def gen_A(self):
        """
            generate A from density n
            in the gauge Ax=0, Ay(x,y)=int_{0}^{x} dxi B(xi,y)
        """
        assert not self.A_updated, "gen_A has been called by update_n, you need not call it manually"
        # A[i,j] i->y, j->x
        B=B0-(m_lauphlin-1)*self.n
        self.Ax=Fqhe.gen_Ax(B)*ax_wt
        self.Ay=Fqhe.gen_Ay(B)*ay_wt
        self.A_updated=True
        #log("Ax: %s, Ay: %s"%(Fqhe.range_on_disk(self.Ax),Fqhe.range_on_disk(self.Ay)))
        #Fqhe.view_A((self.Ax,self.Ay),B)

    def gen_Ay(B):
        Ay=numpy.zeros(B.shape)
        for i,j in itertools.product(range(Npts),range(1,Npts)):
            Ay[i,j]=Ay[i,j-2]+2*Lstep*B[i,j-1]
        for i in range(Npts):
            Ay[i,:]-=numpy.mean(Ay[i,:])
        return Ay

    def gen_Ax(B):
        Ax=numpy.zeros(B.shape)
        for i in range(1,Npts):
            Ax[i,:]=Ax[i-2,:]-2*Lstep*B[i-1,:]
        for j in range(Npts):
            Ax[:,j]-=numpy.mean(Ax[:,j])
        return Ax

    def view_A(Axy,Boriginal):
        """
            for debug, generate B from Ax and Ay then show it
        """
        Ax,Ay=Axy
        B1=numpy.zeros(Ax.shape)
        for j in range(Npts):
            if j>0 and j<Npts-1:
                B1[:,j]-=Ay[:,j-1]
                B1[:,j]+=Ay[:,j+1]
        for i in range(Npts):
            if i>0 and i<Npts-1:
                B1[i,:]+=Ax[i-1,:]
                B1[i,:]-=Ax[i+1,:]
        B1/=(2*Lstep)
        mats=[Boriginal,B1,Ax,Ay]
        titles=['original B','generated by Ax, Ay','Ax','Ay']
        heatmap(mats,titles)

    def gen_VT(self,eigvec):
        """
            generate VT for orbits eigvec given Axy
            epe_last and lr are for slow-updating
        """
        assert self.n_updated, "seems that you are generating VT twice without update n"
        assert self.A_updated, "seems that A has not been updated"
        VTy=Fqhe.gen_VTy(self.Ay,eigvec)
        VTx=Fqhe.gen_VTx(self.Ax,eigvec)
        #heatmap([VTx,VTy],["VTx","VTy"])
        VT=(ay_wt*VTy+ax_wt*VTx)/self.VTrn
        #log("lr: %.4f\nnew VT:\n%s\nlast_VT:\n%s"%(self.lr,VT,self.VT_last))
        if self.VT_reset_flag:
            log("VT reset")
            self.VT_reset_flag=False
            self.VT_last=VT*0.25
        else:
            vtlr=self.lr*2
            vtlr=vtlr if vtlr<1 else self.lr
            self.VT_last=VT*vtlr+self.VT_last*(1-vtlr)
        #log("updated VT:\n%s"%(self.VT_last))
        #heatmap([VT,self.VT_last])
        self.n_updated=False
        return self.VT_last

    def gen_VTy(Ay,eigvec):
        dVT=numpy.zeros(Ay.shape)
        for e in eigvec.transpose():
            e=e.reshape(Ay.shape)
            pe=numpy.zeros(Ay.shape,dtype=numpy.cdouble)
            for i in range(Npts):
                if i>0:
                    pe[i,:]-=e[i-1,:]
                if i<Npts-1:
                    pe[i,:]+=e[i+1,:]
            pe/=(2*Lstep*2*math.pi*1j)
            epe=e.conjugate()*pe
            # eae should equal n*Ay*Lstep**2
            eae=e.conjugate()*Ay*e
            dVT+=epe.real+eae.real
        VT=numpy.zeros(Ay.shape)
        VT[:,Npts-1]=dVT[:,Npts-1]/2
        for j in range(Npts-2,-1,-1):
            VT[:,j]=VT[:,j+1]+(dVT[:,j]+dVT[:,j+1])/2
        wtemp=numpy.array([1-(j+0.5)/Npts for j in range(Npts)])
        VT2=dVT*wtemp
        VT2=numpy.expand_dims(VT2.sum(axis=1),1)
        #heatmap([VT,VT2])
        VT-=VT2
        VT*=(-(m_lauphlin-1)/Mcf)/Lstep
        return VT

    def gen_VTx(Ax,eigvec):
        dVT=numpy.zeros(Ax.shape)
        for e in eigvec.transpose():
            e=e.reshape(Ax.shape)
            pe=numpy.zeros(Ax.shape,dtype=numpy.cdouble)
            for j in range(Npts):
                if j>0:
                    pe[:,j]-=e[:,j-1]
                if j<Npts-1:
                    pe[:,j]+=e[:,j+1]
            pe/=(4*Lstep*math.pi*1j)
            epe=e.conjugate()*pe
            eae=e.conjugate()*Ax*e
            dVT+=epe.real+eae.real
        VT=numpy.zeros(Ax.shape)
        VT[-1,:]=dVT[-1,:]/2
        for i in range(Npts-2,-1,-1):
            VT[i,:]=VT[i+1,:]+(dVT[i,:]+dVT[i+1,:])/2
        VT2=dVT*numpy.expand_dims(numpy.array([1-(i+0.5)/Npts for i in range(Npts)]),1)
        VT2=VT2.sum(axis=0)
        #heatmap([VT,numpy.array([VT2])])
        VT-=VT2
        VT*=((m_lauphlin-1)/Mcf)/Lstep
        return VT

    def gen_Vxc(self):
        """
            generate Vxc for density n
        """
        Vxc=(3/2)*self.a*numpy.sqrt(self.n)+2*(self.b-self.f/2)*self.n+self.g
        return Vxc

    def gen_Ve(self):
        """
            electric potential, including Vext and VH
        """
        n_all=n_posi+self.n
        assert abs(n_all.sum()*Lstep**2)<1e-8,"Sum of positive-negative charges not zero: %.4e"%(n_all.sum())
        Ve=numpy.zeros((Npts,Npts))
        for i,j in itertools.product(range(Npts),range(Npts)):
            Ve+=n_all[i,j]*ewt_tab[Npts-1-i:2*Npts-1-i,Npts-1-j:2*Npts-1-j]
            #assert ewt_tab[Npts-1-i:2*Npts-1-i,Npts-1-j:2*Npts-1-j][i,j]==3.7393513
        Ve*=Lstep/(4*math.pi*ep0)
        return Ve

    def range_on_disk(ax):
        """
            for debug
            return the range of ax on the positive background disk
        """
        bx=ax*(n_posi!=0)
        return "%.2f~%.2f"%(bx.min(),bx.max())

    def tot_charge(n,mask=None):
        if mask is None:
            return n.sum()*Lstep**2
        else:
            return (n*mask).sum()*Lstep**2

    def eig_to_n(eigvec):
        n=eigvec.real**2+eigvec.imag**2
        n=n.sum(axis=1).reshape((Npts,Npts))
        n/=Lstep**2
        assert abs(Fqhe.tot_charge(n)-Ne)<1e-8, "Seems there are too many electrons: %.8f!=%d"%(n.sum()*Lstep**2,Ne)
        return n

    def gen_initst_a():
        """
        generate init state by plane waves
        """
        sN=math.ceil(math.sqrt(Ne)/2)
        l=[(i,j) for i in range(-sN,sN+1) for j in range(-sN,sN+1)]
        l.sort(key=lambda x:abs(x[0])+abs(x[1]))
        log("generating init state: %s"%(l[0:Ne]))
        eigvec=[]
        for i,j in l[0:Ne]:
            e=numpy.exp([math.pi*(i*i2/Npts+j*j2/Npts)*1j for i2 in range(Npts) for j2 in range(Npts)])
            eigvec.append(e)
        eigvec=numpy.transpose(eigvec)/math.sqrt(Npts*Npts)
        return Fqhe.eig_to_n(eigvec),eigvec

    def gen_LL(Lz,num_lls=4):
        """
        generate Landau levels
        """
        r=numpy.zeros(Npts*Npts,dtype=numpy.cdouble)
        for i,j in itertools.product(range(Npts),range(Npts)):
            nft=i*Npts+j
            r[nft]=((i-Npts/2+0.5)+(j-Npts/2+0.5)*1j)*Lstep/lb
        rsq=r.real**2+r.imag**2

        lls=[[] for i in range(num_lls)]
        for m in range(Lz):
            e=(r**m)*numpy.exp(-rsq/4)
            e/=numpy.linalg.norm(e)
            lls[0].append(e)
        for n,m in itertools.product(range(1,num_lls),range(Lz)):
            if n==1:
                e=lls[0][m]*((m+1)-(1/2)*rsq)
            elif n==2:
                e=lls[0][m]*((m+1)*(m+2)-(m+2)*rsq+(1/4)*rsq**2)
            elif n==3:
                e=lls[0][m]*((m+1)*(m+2)*(m+3)-(3/2)*(m+2)*(m+3)*rsq+(3/4)*(m+3)*rsq**2-(1/8)*rsq**3)
            e/=numpy.linalg.norm(e)
            lls[n].append(e)
        return lls

    def gen_initst_b():
        """eigvec=[]
        for m in range(Ne):
            e=numpy.zeros(Npts*Npts,dtype=numpy.cdouble)
            for i,j in itertools.product(range(Npts),range(Npts)):
                r=((i-Npts/2+0.5)+(j-Npts/2+0.5)*1j)*Lstep/lb
                e[i*Npts+j]=(r**m)*math.exp(-abs(r)**2/4)
            e/=numpy.linalg.norm(e)
            eigvec.append(e)"""
        eigvec=Fqhe.gen_LL(Ne,num_lls=1)[0]
        eigvec=numpy.transpose(eigvec)
        return Fqhe.eig_to_n(eigvec),eigvec

    def gen_initst_c():
        LLs=Fqhe.gen_LL(Ne,num_lls=4)
        eigvec=[]
        for m in range(Ne):
            e=LLs[0][m]-LLs[1][m]+LLs[2][m]-LLs[3][m]
            e/=numpy.linalg.norm(e)
            eigvec.append(e)
        eigvec=numpy.transpose(eigvec)
        return Fqhe.eig_to_n(eigvec),eigvec

    def save_state(savename,n_neo,eigvec):
        with open(savename,"wb") as f:
            log("saving state to %s"%(savename))
            pickle.dump((n_neo,eigvec),f)

    def load_initst(filename):
        log("loading init state from %s"%(filename))
        with open(filename,"rb") as f:
            n_neo,eigvec=pickle.load(f)
        return n_neo,eigvec

    def gen_T(self):
        """
            generate kinetic term in Hamiltonian
        """
        assert self.A_updated
        # PP=-\frac{1}{4 \pi^2}\nabla^2
        PP=[];indr=[];indc=[]
        for i,j in itertools.product(range(Npts),range(Npts)):
            nft=i*Npts+j # flattened index n
            PP.append(-4);indr.append(nft);indc.append(nft)
            if j>0:
                PP.append(1);indr.append(nft);indc.append(nft-1)
            if j<Npts-1:
                PP.append(1);indr.append(nft);indc.append(nft+1)
            if i>0:
                PP.append(1);indr.append(nft);indc.append(nft-Npts)
            if i<Npts-1:
                PP.append(1);indr.append(nft);indc.append(nft+Npts)
        PP=sparse.coo_matrix((PP,(indr,indc)),shape=(self.n.size,self.n.size),dtype=numpy.cdouble)/Lstep**2
        PP/=-4*numpy.pi**2

        AP=[];indr=[];indc=[]
        for i,j in itertools.product(range(Npts),range(Npts)):
            nft=i*Npts+j
            if j>0:
                AP.append(-self.Ax[i,j]);indr.append(nft);indc.append(nft-1)
            if j<Npts-1:
                AP.append(self.Ax[i,j]);indr.append(nft);indc.append(nft+1)
            if i>0:
                AP.append(-self.Ay[i,j]);indr.append(nft);indc.append(nft-Npts)
            if i<Npts-1:
                AP.append(self.Ay[i,j]);indr.append(nft);indc.append(nft+Npts)
        AP=sparse.coo_matrix((AP,(indr,indc)),shape=(self.n.size,self.n.size),dtype=numpy.cdouble)/(2*Lstep)
        AP/=(2*numpy.pi*1j)
        AP+=AP.getH()

        AA=[];indr=[];indc=[]
        for i,j in itertools.product(range(Npts),range(Npts)):
            nft=i*Npts+j
            AA.append(self.Ax[i,j]**2+self.Ay[i,j]**2)
            indr.append(nft);indc.append(nft)
        AA=sparse.coo_matrix((AA,(indr,indc)),shape=(self.n.size,self.n.size),dtype=numpy.cdouble)
        return (PP+AP+AA)/(2*Mcf)

    def gen_Vks(self,eigvec):
        """
            generate potential term in Hamiltonian
        """
        assert self.A_updated
        Vxc=self.gen_Vxc()
        if self.VT_switch:
            VT=self.gen_VT(eigvec)
        else:
            VT=numpy.zeros(self.n.shape)
        Ve=self.gen_Ve()
        #Fqhe.range_on_disk(Ve),Fqhe.range_on_disk(Vxc),Fqhe.range_on_disk(VT),Fqhe.range_on_disk(Vimp)
        log("Ve: %s, Vxc: %s, VT: %s, Vimp: %s"%(tuple([Fqhe.range_on_disk(i) for i in [Ve,Vxc,VT,self.Vimp]])))
        #heatmap([self.n,Ve,Vxc,VT],["n","Ve","Vxc","VT"])

        Vks=Vxc+VT+Ve+self.Vimp
        Vks=[Vks[i,j] for i,j in itertools.product(range(Npts),range(Npts))]
        ind=[i*Npts+j for i,j in itertools.product(range(Npts),range(Npts))]
        Vks=sparse.coo_matrix((Vks,(ind,ind)),shape=(self.n.size,self.n.size),dtype=numpy.cdouble)
        return Vks

def dft_step(F,eigvec):
    T=F.gen_T()
    Vks=F.gen_Vks(eigvec)
    H=T+Vks
    assert numpy.abs(H-H.getH()).max()<1e-13, "H not Hermitian: %.4e"%(numpy.abs(H-H.getH()).max())
    max_energy=scipy.sparse.linalg.eigs(H,k=1,return_eigenvectors=False).real
    H-=sparse.eye(F.n.size,dtype=numpy.cdouble)*max_energy[0]
    H=H.tocsr()
    energies,eigvec=scipy.sparse.linalg.eigs(H,k=Ne,v0=eigvec[:,0])
    #log("energies:\n%s"%(" ".join(["%.2f"%(i+max_energy) for i in energies.real])))
    return Fqhe.eig_to_n(eigvec),eigvec

def eval_sym(n):
    twompt=Npts-1
    score=0
    for i,j in itertools.product(range(Npts//2),range(Npts//2)):
        l2=[n[i,j],n[twompt-i,j],n[i,twompt-j],n[twompt-i,twompt-j]]
        score+=max(l2)-min(l2)
    return score/(Npts//2)**2

def eval_sym_updown(n):
    twompt=Npts-1
    score=0
    for i,j in itertools.product(range(Npts//2),range(Npts)):
        l2=[n[i,j],n[twompt-i,j],]
        score+=max(l2)-min(l2)
    return score/((Npts//2)*Npts)

def main():
    figdir="./fig_%s"%(time.strftime("%b%d").lower())
    if not os.path.exists(figdir):
        os.mkdir(figdir)
    saveflag=True
    checkpts=[0.02,0.01,0.002,0.001]

    Vimp=gen_Vimp((Lcut/2,Lcut/2),2*lb,1)
    heatmap([n_posi,Vimp],["posi bk","Vimp"])

    #n_neo,eigvec=Fqhe.gen_initst_c()
    n_neo,eigvec=Fqhe.load_initst("./Ne32Npts200/eig_Ne32_m3_nu33_VTFalse_dN1215.dump")

    #F=Fqhe(n_neo,eigvec,VT_switch=False)
    #F=Fqhe(n_neo,eigvec,VT_switch=True)
    #F=Fqhe(n_neo,eigvec,VT_switch=False,Vimp=Vimp)
    F=Fqhe(n_neo,eigvec,VT_switch=True,Vimp=Vimp)

    #heatmap([n_posi,n_neo],["positive bk","init st"])
    heatmap([n_neo,n_neo],["","init st"],savename=figdir+"/0.png")
    plot_profile(F.n)

    ne_ints=[Fqhe.tot_charge(n_neo,mask_inter),]
    log("charges inside R=%.2f: %.4f"%(Rinter,ne_ints[-1]))
    for i in range(1,41):
        #if i%10==0:
        #    F.VT_reset_flag=True

        n_neo,eigvec=dft_step(F,eigvec)

        dN=numpy.abs(n_neo-F.n).sum()*Lstep**2
        sym_score=eval_sym(n_neo)
        #sym_score=eval_sym_updown(n_neo)
        if sym_score<0.01:
            lr=0.1
        else:
            """sym_score_n=eval_sym(F.n)
            log("sym_scores: %.8f %.8f"%(sym_score,sym_score_n))
            lr=min(sym_score_n/(sym_score+sym_score_n),0.05)"""
            lr=min(dN/Ne+0.01,0.1)
        F.update_n(n_neo,lr)

        log("iter %3d: dN=%.4f, sym=%.4f, lr=%.1f%%"%(i,dN,sym_score,lr*100))
        ne_ints.append(Fqhe.tot_charge(n_neo,mask_inter))
        log("charges inside R=%.2f: %.4f"%(Rinter,ne_ints[-1]))
        heatmap([n_neo,F.n],["neo n, sym_score=%.4f"%(sym_score),"avg n, dN=%.4f"%(dN)],savename=figdir+"/%d.png"%(i))
        plot_profile(F.n)

        for cki,ck in enumerate(checkpts):
            if ck==None:
                continue
            elif dN<Ne*ck:
                if saveflag:
                    savename="eig_Ne%d_m%d_nu%d_VT%s_dN%04d.dump"%(Ne,m_lauphlin,nu*1e2,F.VT_switch,dN*1e4)
                    Fqhe.save_state(savename,n_neo,eigvec)
                checkpts[cki]=None
            else:
                break
        else:
            log("stop at iter %d"%(i))
            break

    plot_profile(F.n)
    log(",".join(["%.4f"%(i) for i in ne_ints]))
    if saveflag:
        savename="eig_Ne%d_m%d_nu%d_VT%s_dN%04d.dump"%(Ne,m_lauphlin,nu*1e2,F.VT_switch,dN*1e4)
        Fqhe.save_state(savename,n_neo,eigvec)

def plot1():
    n_neo,eigvec=Fqhe.load_initst("./Ne32Npts200/eig_Ne32_m3_nu33_VTTrue_dN3062.dump")
    heatmap([n_neo,n_posi,mask_inter])
    log("inter charges: %.4f"%(Fqhe.tot_charge(n_neo,mask_inter)))
    plot_profile(n_neo)
    """F=Fqhe(n_neo,eigvec,VT_switch=True)
    F.gen_Vks(eigvec)"""

def plot2():
    #n_neo,eigvec=Fqhe.load_initst("./Ne32Npts200/eig_Ne32_m3_nu33_VTTrue_dN5216.dump")
    n_neo,eigvec=Fqhe.load_initst("./Ne10Npts128/eig_VTTrue_m3_nu33_dN0089.dump")

    num_lls=4
    lls=Fqhe.gen_LL(30,num_lls=num_lls)
    eigvec=eigvec.transpose()
    all_ll=[[] for i in eigvec]

    xs=list(range(len(lls[0])))
    fig,axs=plt.subplots(1,num_lls,figsize=(6.4*num_lls,4.8))
    for nl in range(num_lls):
        for i,e in enumerate(eigvec):
            this_ll=[abs(numpy.vdot(e,l)) for l in lls[nl]]
            axs[nl].plot(xs,this_ll,label="%d"%(i))
            all_ll[i]+=this_ll
        log(["%.4f"%(sum([abs(j)**2 for j in i])) for i in all_ll])
        axs[nl].set_ylim(0,0.8)
    axs[0].legend()
    plt.show()

if __name__=="__main__":
    main()