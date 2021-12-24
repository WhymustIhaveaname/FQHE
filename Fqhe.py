#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time,sys,traceback,math

LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE="/home/public/youran/Fqhe.log"
def log(msg,l=1,end="\n",logfile=LOGFILE):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    #now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    now_str="%s"%(time.strftime("%y/%b %H:%M:%S",time.localtime()),)
    perfix="%s [%s,%s:%03d]"%(now_str,lstr,st.name,st.lineno)
    if l<3:
        tempstr="%s %s%s"%(perfix,str(msg),end)
    else:
        tempstr="%s %s:\n%s%s"%(perfix,str(msg),traceback.format_exc(limit=5),end)
    print(tempstr,end="")
    if l>=1:
        with open(logfile,"a") as f:
            f.write(tempstr)

import math,numpy,itertools,scipy,scipy.linalg,scipy.sparse.linalg,seaborn
from scipy import sparse
import matplotlib.pyplot as plt

# Number of electrons
Ne=10
# Laughlin m
m_lauphlin=3
# Filling fraction
nu=1/m_lauphlin
# mass for CF, under the nature unit me=1
Mcf=0.067
# Sample Radius
Rsmp=math.sqrt(Ne/(nu*math.pi))
# magnetic length under this unit
lb=1/math.sqrt(2*math.pi)
log("Sample radius when Ne=%d, \\nu=%.4f: %.4f"%(Ne,nu,Rsmp))
# Compute boundry size
Lcut=2*(Rsmp+1*lb)
#Lcut=math.sqrt(Ne/nu)
# Discrete number and step size in each direction
Npts=128
Lstep=Lcut/Npts
log("m, nu, Mcf: %d, %.3f, %.3f"%(m_lauphlin,nu,Mcf))
log("Cut Length, discrete step size: %.4f, %.4f"%(Lcut,Lstep))
# weight for Ay
ax_wt=0.0
ay_wt=1.0
log("ax_wt, ay_wt: %.1f, %.1f"%(ax_wt,ay_wt))
# vacuum permittivity
# using the data for GaAs
ep0=2.585105102e-3*12.6
# background B
B0=1


# positive disk
sq2=math.sqrt(2)
def calc_nposi(i,j):
    rij=math.sqrt((i+0.5-Npts/2)**2+(j+0.5-Npts/2)**2)
    dr=(Rsmp/Lstep)-(rij-sq2/2)
    if dr>sq2:
        return 1
    elif dr<0:
        return 0
    elif dr<sq2/2:
        return dr**2
    else:
        return 1-(sq2-dr)**2

n_posi=numpy.zeros((Npts,Npts))
for i,j in itertools.product(range(Npts),range(Npts)):
    n_posi[i,j]=calc_nposi(i,j)
n_posi*=-Ne/(n_posi.sum()*Lstep**2)
del sq2

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

class Fqhe():
    def __init__(self,n):
        self.n=n
        self.VT_last=None
        self.update_n(n,1)

        # some parameters from Hu PRL 123, 176802 (2019)
        # will be used in gen_Vxc
        eu=1/(4*math.pi*ep0*lb)
        self.a=-0.78213*eu;self.b=0.2774*eu;self.f=0.33*eu;self.g=-0.04981*eu
        log("eu, a, b, f: %.4f, %.2f, %.2f, %.2f"%(eu,self.a,self.b,self.f))

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
        # A[i,j] i->y, j->x
        B=B0-(m_lauphlin-1)*self.n
        self.Ax=Fqhe.gen_Ax(B)*ax_wt
        self.Ay=Fqhe.gen_Ay(B)*ay_wt
        self.A_updated=True
        log("Ax: %s, Ay: %s"%(Fqhe.range_on_disk(self.Ax),Fqhe.range_on_disk(self.Ay)))

    def gen_Ay(B):
        Ay=numpy.zeros(B.shape)
        for i,j in itertools.product(range(Npts),range(1,Npts)):
            Ay[i,j]=Ay[i,j-2]+2*Lstep*B[i,j-1]
        for i in range(Npts):
            Ay[i,:]-=numpy.mean(Ay[i,:])
        return Ay

    def gen_Ax(B):
        Ax=numpy.zeros(B.shape)
        for i,j in itertools.product(range(1,Npts),range(Npts)):
            Ax[i,:]=Ax[i-2,:]-2*Lstep*B[i-1,:]
        for j in range(Npts):
            Ax[:,j]-=numpy.mean(Ax[:,j])
        return Ax

    def gen_VT(self,eigvec):
        """
            generate VT for orbits eigvec given Axy
            epe_last and lr are for slow-updating
        """
        VTy=Fqhe.gen_VTy(self.Ay,eigvec)
        VTx=Fqhe.gen_VTx(self.Ax,eigvec)
        VT=ay_wt*VTy+ax_wt*VTx
        # maybe?
        #VT/=Ne
        #VT/=2

        if self.VT_last is not None:
            VTre=VT*self.lr+self.VT_last*(1-self.lr)
        else:
            VTre=VT

        if self.n_updated:
            self.VT_last=VT
            self.n_updated=False
        else:
            log("Trying to compute VT before update n?",l=2)

        return VTre

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
        VT2=dVT*numpy.array([1-(j+0.5)/Npts for j in range(Npts)])
        VT-=numpy.expand_dims(VT2.sum(axis=1),1)
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
        bx=ax*(n_posi!=0)
        return "%.2f~%.2f"%(bx.min(),bx.max())

    def eig_to_n(eigvec):
        n=eigvec.real**2+eigvec.imag**2
        n=n.sum(axis=1).reshape((Npts,Npts))
        n/=Lstep**2
        assert abs(n.sum()*Lstep**2-Ne)<1e-8, "Seems there are too many electrons: %.8f!=%d"%(n.sum()*Lstep**2,Ne)
        return n

    def gen_initst_a():
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

    def gen_initst_b():
        eigvec=[]
        for m in range(Ne):
            e=numpy.zeros(Npts*Npts,dtype=numpy.cdouble)
            for i,j in itertools.product(range(Npts),range(Npts)):
                r=((i-Npts/2+0.5)+(j-Npts/2+0.5)*1j)*Lstep/lb
                e[i*Npts+j]=(r**m)*math.exp(-abs(r)**2/4)
            e/=numpy.linalg.norm(e)
            eigvec.append(e)
        eigvec=numpy.transpose(eigvec)
        return Fqhe.eig_to_n(eigvec),eigvec

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
        VT=self.gen_VT(eigvec)
        #VT=numpy.zeros(self.n.shape)
        Ve=self.gen_Ve()
        log("Ve: %s, Vxc: %s, VT: %s"%(Fqhe.range_on_disk(Ve),Fqhe.range_on_disk(Vxc),Fqhe.range_on_disk(VT)))

        Vks=Vxc+VT+Ve
        Vks=[Vks[i,j] for i,j in itertools.product(range(Npts),range(Npts))]
        ind=[i*Npts+j for i,j in itertools.product(range(Npts),range(Npts))]
        Vks=sparse.coo_matrix((Vks,(ind,ind)),shape=(self.n.size,self.n.size),dtype=numpy.cdouble)
        return Vks

def test1():
    n,eigvec=Fqhe.gen_initst_a()
    F=Fqhe(n)
    #log("density n:\n%s"%(n)) #pass
    log("Ay\n%s"%(F.Ay)) #pass
    Vxc=F.gen_Vxc()
    #log("Vxc\n%s"%(Vxc)) #pass
    VT=F.gen_VT(eigvec)
    #log("VT\n%s"%(VT))
    Ve=F.gen_Ve()
    log("Ve: %.4f~%.4f"%(Ve.min(),Ve.max()))
    seaborn.heatmap(Ve,square=True)
    plt.show()

def dft_step(F,eigvec):
    T=F.gen_T()
    Vks=F.gen_Vks(eigvec)
    H=T+Vks
    assert numpy.abs(H-H.getH()).max()<1e-13, "H not Hermitian: %.4e"%(numpy.abs(H-H.getH()).max())
    max_energy=scipy.sparse.linalg.eigs(H,k=1,return_eigenvectors=False).real
    H-=sparse.eye(F.n.size,dtype=numpy.cdouble)*max_energy[0]
    H=H.tocsr()
    energies,eigvec=scipy.sparse.linalg.eigs(H,k=Ne)
    log("energies: %s"%(energies.real[0:4]+max_energy))
    return Fqhe.eig_to_n(eigvec),eigvec

def main(N_update_per_step=0.5):
    """log("n_posi.sum(): %.2f=-%d+%e"%(n_posi.sum()*Lstep**2,Ne,n_posi.sum()*Lstep**2+Ne))
    seaborn.heatmap(n_posi,square=True,xticklabels=False, yticklabels=False)
    plt.show()"""
    n_neo,eigvec=Fqhe.gen_initst_b()
    F=Fqhe(n_neo)
    for i in range(31):
        #if i<5 or i%5==0:
        if True:
            """fig,(ax1,ax2)=plt.subplots(1,2)
            seaborn.heatmap(F.Ax,square=True,xticklabels=False,yticklabels=False,ax=ax1)
            seaborn.heatmap(F.Ay,square=True,xticklabels=False,yticklabels=False,ax=ax2)
            plt.show()"""
            fig,(ax1,ax2)=plt.subplots(1,2)
            seaborn.heatmap(n_neo,square=True,ax=ax1,xticklabels=False, yticklabels=False)
            seaborn.heatmap(F.n,square=True,ax=ax2,xticklabels=False, yticklabels=False)
            #fig.savefig("./visiter_dec19/%d.png"%(i))
            plt.show()

        n_neo,eigvec=dft_step(F,eigvec)
        dN=numpy.abs(n_neo-F.n).sum()*Lstep**2
        lr=min(0.2,N_update_per_step/dN)
        log("iter %3d: dN=%.4f, lr=%.3f"%(i,dN,lr))
        F.update_n(n_neo,lr)
        n_last=n_neo
    seaborn.heatmap(F.n,square=True,xticklabels=False, yticklabels=False)
    plt.show()

if __name__=="__main__":
    #test1()
    main()