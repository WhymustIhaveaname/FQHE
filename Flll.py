#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Fqhe import *
import scipy.linalg

class Flll(Fqhe):
    def __init__(self,n,VT_switch=False,VTrn=1.0,Vimp=None,Lz_ext=4):
        Fqhe.__init__(self,n,VT_switch=VT_switch,VTrn=VTrn,Vimp=Vimp)
        self.Lz_max=int(Ne/nu)+Lz_ext
        self.LLL=Futil.gen_LL(self.Lz_max,num_lls=1)[0]

def dft_step_lll(F,eigvec):
    log("generating sparse Hamiltonian")
    T=F.gen_T()
    Vks=F.gen_Vks(eigvec)
    H=(T+Vks).tocsr()
    diff_max=numpy.abs(H-H.getH()).max()
    assert diff_max<1e-15, "H not Hermitian: %.4e"%(diff_max)

    log("generating projected Hamiltonian")
    Hlll=numpy.zeros((F.Lz_max,F.Lz_max),dtype=numpy.cdouble)
    for i in range(F.Lz_max):
        Hvi=H.dot(F.LLL[i])
        Hlll[i,i]=numpy.vdot(F.LLL[i],Hvi)
        for j in range(i):
            Hlll[j,i]=numpy.vdot(F.LLL[j],Hvi)
            Hlll[i,j]=numpy.conjugate(Hlll[j,i])
    diff_max=numpy.abs(Hlll-Hlll.transpose().conjugate()).max()
    assert diff_max<1e-14, "Hlll not Hermitian: %.4e"%(diff_max)

    energies,eigvec=scipy.linalg.eigh(Hlll)
    print(energies)

def test_dft_step_lll():
    n_neo,eigvec=Futil.gen_initst_b()
    F=Flll(n_neo,VT_switch=True)

    Futil.heatmap([n_neo,],["init st",])
    Futil.plot_profile(F.n)

    dft_step_lll(F,eigvec)