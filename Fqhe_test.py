#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Fqhe import *

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

def test_gen_Vimp():
    hs=(1.0,2.0,3.0,4.0)
    Vimps=[gen_Vimp((Lcut/2,Lcut/2),h,1) for h in hs]
    heatmap(Vimps,["h=%1.f"%(h) for h in hs])
    #plot_profile(Vimps[0])

if __name__=="__main__":
    test1()
    test_gen_Vimp()