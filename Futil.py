#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Fconsts import *
import itertools,pickle
import matplotlib.pyplot as plt

class Futil():
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

    def range_on_disk(ax):
        """
            for debug
            return the range of ax on the positive background disk
        """
        bx=ax*(n_posi!=0)
        return "%.2f~%.2f"%(bx.min(),bx.max())

    def count_charges(n,mask=None):
        """
            if mask is None, return all charges
            else return \\int mask*n
        """
        if mask is None:
            return n.sum()*Lstep**2
        else:
            ch=(n*mask).sum()*Lstep**2
            log("interior charges: %.4f"%(ch))
            return ch

    """
        functions about generating init states
    """

    def eig_to_n(eigvec):
        n=eigvec.real**2+eigvec.imag**2
        n=n.sum(axis=1).reshape((Npts,Npts))
        n/=Lstep**2
        assert abs(Futil.count_charges(n)-Ne)<1e-8, "Seems there are too many electrons: %.8f!=%d"%(n.sum()*Lstep**2,Ne)
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

    """
        save and load
    """

    def save_state(savename,n_neo,eigvec):
        with open(savename,"wb") as f:
            log("saving state to %s"%(savename))
            pickle.dump((n_neo,eigvec),f)

    def load_initst(filename):
        log("loading init state from %s"%(filename))
        with open(filename,"rb") as f:
            n_neo,eigvec=pickle.load(f)
        return n_neo,eigvec

    """
        functions for visualize
    """

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

def plot1():
    n_neo,eigvec=Futil.load_initst("./Ne32Npts200/eig_Ne32_m3_nu33_VTTrue_dN3062.dump")
    heatmap([n_neo,n_posi,mask_inter])
    log("inter charges: %.4f"%(Fqhe.count_charges(n_neo,mask_inter)))
    plot_profile(n_neo)

def plot2():
    n_neo,eigvec=Futil.load_initst("./Ne10Npts128/eig_VTTrue_m3_nu33_dN0089.dump")

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