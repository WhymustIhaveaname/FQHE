#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Fconsts import *
from Futil import *
from Fqhe import Fqhe,figdir

def Flll(Fqhe):
    def __init__(self,n,VT_switch=False,VTrn=1.0,Vimp=None):
        self.n=n
        self.VT_switch=VT_switch
        if Vimp is not None:
            self.Vimp=Vimp
        else:
            self.Vimp=numpy.zeros(self.n.shape)
        log("VT: %s, VT_renormalization: %.1f"%(self.VT_switch,VTrn))

        self.LLL=Futil.gen_LL(int(Ne/nu))[0]

        self.update_n(n,1)
        if self.VT_switch:
            self.VTrn=VTrn
            self.VT_reset_flag=False
            self.VT_last=numpy.zeros(self.n.shape)

def dft_step_lll(F):
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