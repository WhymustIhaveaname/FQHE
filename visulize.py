#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import imageio
from PIL import Image,ImageDraw,ImageFont

def to_gif(pattern,iter,end_dup,dur,save_name):
    frames=[]
    myfont=ImageFont.truetype("./Ubuntu_Mono/UbuntuMono-Regular.ttf",size=24)
    for i in iter:
        print(i)
        img=imageio.imread(pattern%(i))
        img=Image.fromarray(img,"RGBA")
        imgdraw=ImageDraw.Draw(img)
        imgdraw.text((16,16),"iter %2d"%(i),fill=(0,0,0),font=myfont)
        frames.append(img)
    for i in range(end_dup):
        frames.append(frames[-1])
        frames.insert(0,frames[0])
    imageio.mimsave(save_name,frames,'GIF',duration=dur)

if __name__=="__main__":
    working_dir="./visiter_Vy_dec19/"
    to_gif(working_dir+"%d.png",range(51),3,0.5,working_dir+"visiter.gif")