#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math,numpy,itertools,os
import scipy,scipy.sparse.linalg
from scipy import sparse
from Fconsts import *
from Futil import *

figdir="./fig_%s"%(time.strftime("%b%d").lower())
if not os.path.exists(figdir):
    os.mkdir(figdir)

class Fqhe():
    def __init__(self,n,eigvec,VT_switch=False,VTrn=1.0,Vimp=None):
        self.n=n
        self.VT_switch=VT_switch
        if Vimp is not None:
            self.Vimp=Vimp
        else:
            self.Vimp=numpy.zeros(self.n.shape)
        log("VT: %s, VT_renormalization: %.1f"%(self.VT_switch,VTrn))

        self.update_n(n,1)
        if self.VT_switch:
            self.VTrn=VTrn
            self.VT_reset_flag=False
            self.VT_last=numpy.zeros(self.n.shape)

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

    def gen_VT(self,eigvec):
        """
            generate VT for orbits eigvec given Axy
            epe_last and lr are for slow-updating
        """
        assert self.n_updated, "seems that you are generating VT twice without update n"
        assert self.A_updated, "seems that A has not been updated"
        VTy=Fqhe.gen_VTy(self.Ay,eigvec)
        VTx=Fqhe.gen_VTx(self.Ax,eigvec)
        VT=(ay_wt*VTy+ax_wt*VTx)/self.VTrn

        if self.VT_reset_flag:
            log("VT reset")
            self.VT_reset_flag=False
            self.VT_last=VT
        else:
            #vtlr=self.lr*2
            #vtlr=vtlr if vtlr<1 else self.lr
            vtlr=self.lr
            self.VT_last=VT*vtlr+self.VT_last*(1-vtlr)

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
        # eu = energy unit
        eu=1/(4*math.pi*ep0*lb)
        # some parameters from Hu PRL 123, 176802 (2019)
        a=-0.78213*eu;b=0.2774*eu;f=0.33*eu;g=-0.04981*eu
        Vxc=(3/2)*a*numpy.sqrt(self.n)+2*(b-f/2)*self.n+g
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
        Ve=self.gen_Ve()
        Vxc=self.gen_Vxc()
        if self.VT_switch:
            VT=self.gen_VT(eigvec)
        else:
            VT=numpy.zeros(self.n.shape)

        log("Ve: %s, Vxc: %s, VT: %s, Vimp: %s"%(tuple([Futil.range_on_disk(i) for i in [Ve,Vxc,VT,self.Vimp]])))
        Vks=Ve+Vxc+VT+self.Vimp
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
    return Futil.eig_to_n(eigvec),eigvec

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
    saveflag=True
    checkpts=[0.02,0.01,0.002,0.001]

    Vimp=Futil.gen_Vimp((Lcut/2,Lcut/2),2*lb,1)
    Futil.heatmap([n_posi,Vimp],["posi bk","Vimp"])

    #n_neo,eigvec=Futil.gen_initst_c()
    n_neo,eigvec=Futil.load_initst("./Ne32Npts200/eig_Ne32_m3_nu33_VTFalse_dN1215.dump")

    #F=Fqhe(n_neo,eigvec,VT_switch=False)
    #F=Fqhe(n_neo,eigvec,VT_switch=True)
    #F=Fqhe(n_neo,eigvec,VT_switch=False,Vimp=Vimp)
    F=Fqhe(n_neo,eigvec,VT_switch=True,Vimp=Vimp)

    Futil.heatmap([n_neo,n_neo],["","init st"],savename=figdir+"/0.png")
    Futil.plot_profile(F.n)

    ne_ints=[Futil.count_charges(n_neo,mask_inter),]
    for i in range(1,3):
        #if i%10==0:
        #    F.VT_reset_flag=True

        n_neo,eigvec=dft_step(F,eigvec)

        dN=numpy.abs(n_neo-F.n).sum()*Lstep**2
        sym_score=eval_sym(n_neo)
        #sym_score=eval_sym_updown(n_neo)
        if sym_score<0.01:
            lr=0.1
        else:
            lr=min(dN/Ne+0.01,0.1)
        F.update_n(n_neo,lr)

        log("iter %3d: dN=%.4f, sym=%.4f, lr=%.1f%%"%(i,dN,sym_score,lr*100))
        ne_ints.append(Futil.count_charges(n_neo,mask_inter))
        Futil.heatmap([n_neo,F.n],["neo n, sym_score=%.4f"%(sym_score),"avg n, dN=%.4f"%(dN)],savename=figdir+"/%d.png"%(i))
        Futil.plot_profile(F.n)

        for cki,ck in enumerate(checkpts):
            if ck==None:
                continue
            elif dN<Ne*ck:
                if saveflag:
                    savename="eig_Ne%d_m%d_nu%d_VT%s_dN%04d.dump"%(Ne,m_lauphlin,nu*1e2,F.VT_switch,dN*1e4)
                    Futil.save_state(savename,n_neo,eigvec)
                checkpts[cki]=None
            else:
                break
        else:
            log("stop at iter %d"%(i))
            break

    Futil.plot_profile(F.n)
    log(",".join(["%.4f"%(i) for i in ne_ints]))
    if saveflag:
        savename="eig_Ne%d_m%d_nu%d_VT%s_dN%04d.dump"%(Ne,m_lauphlin,nu*1e2,F.VT_switch,dN*1e4)
        Futil.save_state(savename,n_neo,eigvec)

if __name__=="__main__":
    pass