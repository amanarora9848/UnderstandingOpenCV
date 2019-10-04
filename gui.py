# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:35:04 2019

@author: APPED
"""
#import os
import tkinter as tk
from PIL import Image,ImageTk
import cv2
from skimage.util import random_noise
from functools import partial
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.filters import roberts, sobel,  prewitt,unsharp_mask
import numpy as np


#For contrast purpose
def cont(img,x,enhance):
    cd=x.get()
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for z in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,z,c] = np.clip(cd*img[y,z,c], 0, 255)
    im = Image.fromarray(new_image)
    imgtk = ImageTk.PhotoImage(image=im,master=enhance)
    imglabel=tk.Label(enhance,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)
#for power law
def power(gamma,img,enhance):
    g=float(gamma.get())
    gamma_v = np.array(255*(img/255)**g,dtype='uint8')
    im = Image.fromarray(gamma_v)
    imgtk = ImageTk.PhotoImage(image=im,master=enhance)
    imglabel=tk.Label(enhance,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)    
    #for brightness
def brightness(bright,img,enhance):
    y=bright.get()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    if(y>=0):
        lim = 255 - y
        v[v > lim] = 255
        v[v <= lim] += y
    else:
        lim = -y
        v[v < lim] = 0
        v[v >= lim] =v[v >= lim]- lim
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im,master=enhance)
    imglabel=tk.Label(enhance,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)
    
#for fourier transform
def fourier1(img,enhance):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    f_abs = np.abs(f_complex) + 1
    f_bounded = 20 * np.log(f_abs)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    r = 10
    ham = np.hamming(400)[:,None]
    ham2d = np.sqrt(np.dot(ham, ham.T)) ** r
    f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img*255 / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)
    im = Image.fromarray(filtered_img)
    
    imgtk = ImageTk.PhotoImage(image=im,master=enhance)
    imglabel=tk.Label(enhance,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)
        
 #For smoothing   
def smooth1(img,enhance,x):
    y=x.get()
    if(y==1):
        noise=np.ones((3,3),np.float32)/9
        img=cv2.filter2D(img,-1,noise)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
    elif(y==2):
         noise=np.ones((5,5),np.float32)/25
         img=cv2.filter2D(img,-1,noise)
         im = Image.fromarray(img)
         imgtk = ImageTk.PhotoImage(image=im,master=enhance)
         imglabel=tk.Label(enhance,image=imgtk)
         imglabel.image=imgtk
         imglabel.place(x=100,y=150)
    elif(y==3):
         noise=np.ones((7,7),np.float32)/49
         img=cv2.filter2D(img,-1,noise)
         im = Image.fromarray(img)
         imgtk = ImageTk.PhotoImage(image=im,master=enhance)
         imglabel=tk.Label(enhance,image=imgtk)
         imglabel.image=imgtk
         imglabel.place(x=100,y=150)
    elif(y==4):
        img=cv2.medianBlur(img,15)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
    elif(y==5):
        img=cv2.GaussianBlur(img,(15,15),0)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
            
    #for point transformations    
def point1(img,enhance,x):
    y=x.get()
    if(y==1):
        img=cv2.bitwise_not(img)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
    elif(y==3):
         img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255
         img_log = np.array(img_log,dtype=np.uint8)
         im = Image.fromarray(img_log)
         imgtk = ImageTk.PhotoImage(image=im,master=enhance)
         imglabel=tk.Label(enhance,image=imgtk)
         imglabel.image=imgtk
         imglabel.place(x=100,y=150)
    elif(y==5):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        noise=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
        im = Image.fromarray(noise)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)

#for sharpening
def sharp1(img,enhance,x):
    y=x.get()

    if(y==1):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        lap=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
        im = Image.fromarray(lap)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
    elif(y==2):
         img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
         n=img_as_float(img)
         robert=roberts(n)
         noise=img_as_ubyte(robert)
         im = Image.fromarray(noise)
         imgtk = ImageTk.PhotoImage(image=im,master=enhance)
         imglabel=tk.Label(enhance,image=imgtk)
         imglabel.image=imgtk
         imglabel.place(x=100,y=150)
    elif(y==3):
         img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
         n=img_as_float(img)
         sobe=sobel(n)
         noise=img_as_ubyte(sobe)
         im = Image.fromarray(noise)
         imgtk = ImageTk.PhotoImage(image=im,master=enhance)
         imglabel=tk.Label(enhance,image=imgtk)
         imglabel.image=imgtk
         imglabel.place(x=100,y=150)
    elif(y==4):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img,100,200)
        im = Image.fromarray(edges)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)    
    elif(y==5):
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        n=img_as_float(img)
        pre=prewitt(n)
        noise=img_as_ubyte(pre)
        im = Image.fromarray(noise)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)    
    elif(y==6):
        n=img_as_float(img)
        ma=unsharp_mask(n,radius=2,amount=2)
        noise=img_as_ubyte(ma)
        im = Image.fromarray(noise)
        imgtk = ImageTk.PhotoImage(image=im,master=enhance)
        imglabel=tk.Label(enhance,image=imgtk)
        imglabel.image=imgtk
        imglabel.place(x=100,y=150)
    #for adding noise
def Addnoise(x,img,restore):
    y=x.get()
    if(y==1):
        gauss = np.random.normal(0,0.5,img.size)
        gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        img_gauss = cv2.add(img,gauss)
        im = Image.fromarray(img_gauss)
        imgtk = ImageTk.PhotoImage(image=im,master=restore)
        imgl=tk.Label(restore,image=imgtk)
        imgl.image=imgtk
        imgl.place(x=100,y=150)
    elif(y==2):
        noise_img = random_noise(img, mode='poisson',seed=42,clip=True)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        im = Image.fromarray(noise_img)
        imgtk = ImageTk.PhotoImage(image=im,master=restore)
        imgl=tk.Label(restore,image=imgtk)
        imgl.image=imgtk
        imgl.place(x=100,y=150)
    elif(y==3):
        speckle = np.random.normal(0,0.5,img.size)
        speckle = speckle.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
        img_gauss = img+speckle*img
        im=Image.fromarray(img_gauss)
        imgtk = ImageTk.PhotoImage(image=im,master=restore)
        imgl=tk.Label(restore,image=imgtk)
        imgl.image=imgtk
        imgl.place(x=100,y=150)
    elif(y==4):
        noise_img = random_noise(img, mode='s&p',amount=0.1,salt_vs_pepper=0.3)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        im = Image.fromarray(noise_img)
        imgtk = ImageTk.PhotoImage(image=im,master=restore)
        imgl=tk.Label(restore,image=imgtk)
        imgl.image=imgtk
        imgl.place(x=100,y=150)
    else:
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im,master=restore)
        imgl=tk.Label(restore,image=imgtk)
        imgl.image=imgtk
        imgl.place(x=100,y=150)
 #for reset button   
def reset(img,win):
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im,master=win)
    imglabel=tk.Label(win,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)
    
  #for destroying window  
def back(win):
    win.destroy()
    
#for restoration
def restoration(img):
    restore=tk.Tk()
    x=tk.IntVar(restore)
    restore.title("Image Restoration")
    restore.configure(bg='#447744')
    restore.geometry('1000x700')    
    label=tk.Label(restore,text="Image Restoration",font=("Arial Bold",20),fg="black",bg='#447744')
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im,master=restore)
    label1=tk.Label(restore,text="Select the desired noise:",font=("Arial Bold",15),fg="blue",bg='#447744')
    ls=tk.Radiobutton(restore,text="Gaussian",activebackground="#004400",variable=x,value=1,indicatoron = 0,width = 20,padx = 20,command=partial(Addnoise,x,img,restore))
    ls1=tk.Radiobutton(restore,text="Poisson",activebackground="#004400",variable=x,value=2,indicatoron = 0,width = 20,padx = 20,command=partial(Addnoise,x,img,restore)) 
    ls2=tk.Radiobutton(restore,text="Speckle",activebackground="#004400",variable=x,value=3,indicatoron = 0,width = 20,padx = 20,command=partial(Addnoise,x,img,restore))
    ls3=tk.Radiobutton(restore,text="Salt and pepper",activebackground="#004400",variable=x,value=4,indicatoron = 0,width = 20,padx = 20,command=partial(Addnoise,x,img,restore))
    imgl=tk.Label(restore,image=imgtk)
    imgl.image=imgtk
    imgl.place(x=100,y=150)
    bt=tk.Button(restore,text="Back",activebackground='#447744',font=("Arial Bold",15),fg='black',command=partial(back,restore))
    bt1=tk.Button(restore,text="Reset",activebackground='#447744',font=("Arial Bold",15),fg='black',command=partial(reset,img,restore))
    label.place(x=350,y=50)
    label1.place(x=550,y=100)
    ls.place(x=550,y=150)
    ls1.place(x=550,y=180)
    ls2.place(x=550,y=210)
    ls3.place(x=550,y=240)
    bt.place(x=750,y=150)
    bt1.place(x=750,y=200) 
    restore.mainloop()

    #for enhancement
def enhancement(img):
    enhance=tk.Tk()
    x=tk.IntVar(enhance)
    y1=tk.DoubleVar(enhance)
    z=tk.IntVar(enhance)
    text=tk.StringVar(enhance)
    
    
    enhance.title("Image Enhancement")
    enhance.configure(bg='#447744')
    enhance.geometry('1500x800')    
    
    label=tk.Label(enhance,text="Image Enhancement",font=("Arial Bold",20),fg="black",bg='#447744')
    flabel=tk.Label(enhance,text="Frequency Domain",font=("Arial Bold",15),fg="black",bg='#447744')
    slabel=tk.Label(enhance,text="Spatial Domain",font=("Arial Bold",15),fg="black",bg='#447744')

    sl=tk.Scale(enhance,orient="horizontal",variable=y1,from_=0.1,to=3.0,resolution=0.01)
    sl.place(x=100,y=570)
    contrast=tk.Label(enhance,text="Contrast Value ",font=("Arial Bold",10),fg="black",bg='#447744')
    contrast.place(x=250,y=590)
    
    sl1=tk.Scale(enhance,orient="horizontal",variable=z,from_=-127,to=127)
    sl1.place(x=100,y=650)
    bright=tk.Label(enhance,text="Brightness Value ",font=("Arial Bold",10),fg="black",bg='#447744')
    bright.place(x=250,y=670) 
    
    ytext=tk.Entry(enhance,bd=5,textvariable=text,width=10)
    ytext.place(x=100,y=750)
    gamma=tk.Label(enhance,text="Enter Gamma ",font=("Arial Bold",10),fg="black",bg='#447744')
    gamma.place(x=200,y=750)            
    
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im,master=enhance)
    imglabel=tk.Label(enhance,image=imgtk)
    imglabel.image=imgtk
    imglabel.place(x=100,y=150)
    
    pointbt=tk.Label(enhance,text="Point Processing",bg='#447744',font=("Arial Bold",15),fg='black')
    smoothbt=tk.Label(enhance,text="Smoothing",bg='#447744',font=("Arial Bold",15),fg='black')
    sharpbt=tk.Label(enhance,text="Sharpening",bg='#447744',font=("Arial Bold",15),fg='black')
    bt=tk.Button(enhance,text="Back",activebackground='#447744',font=("Arial Bold",15),fg='black',command=partial(back,enhance))
    bt1=tk.Button(enhance,text="Reset",activebackground='#447744',font=("Arial Bold",15),fg='black',command=partial(reset,img,enhance))
    fourierbt=tk.Label(enhance,text="Fourier",bg='#447744',font=("Arial Bold",15),fg='black')
    label.place(x=650,y=50)
    flabel.place(x=550,y=150)
    slabel.place(x=750,y=150)
    bt.place(x=950,y=110)
    bt1.place(x=1050,y=110)
    fourierbt.place(x=550,y=200)
    sharpbt.place(x=750,y=200)
    smoothbt.place(x=900,y=200)
    pointbt.place(x=1050,y=200)
    
    ls=tk.Radiobutton(enhance,text="Laplacian Filter",activebackground="#004400",variable=x,value=1,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x))
    ls1=tk.Radiobutton(enhance,text="Robert Edge Detection",activebackground="#004400",variable=x,value=2,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x)) 
    ls2=tk.Radiobutton(enhance,text="Sobel Filter",activebackground="#004400",variable=x,value=3,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x))
    ls3=tk.Radiobutton(enhance,text="Canny Edge Detection",activebackground="#004400",variable=x,value=4,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x))
    ls4=tk.Radiobutton(enhance,text="Prewitt Filter",activebackground="#004400",variable=x,value=5,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x))
    ls5=tk.Radiobutton(enhance,text="Unsharp Masking",activebackground="#004400",variable=x,value=6,indicatoron = 0,width = 15,padx = 15,command=partial(sharp1,img,enhance,x))
    ls.place(x=750,y=250)
    ls1.place(x=750,y=280)
    ls2.place(x=750,y=310)
    ls3.place(x=750,y=340)
    ls4.place(x=750,y=370)
    ls5.place(x=750,y=400)
    
    ls6=tk.Radiobutton(enhance,text="Average Blur 3x3",activebackground="#004400",variable=x,value=1,indicatoron = 0,width = 15,padx = 15,command=partial(smooth1,img,enhance,x))
    ls7=tk.Radiobutton(enhance,text="Average Blur 5x5",activebackground="#004400",variable=x,value=2,indicatoron = 0,width = 15,padx = 15,command=partial(smooth1,img,enhance,x)) 
    ls8=tk.Radiobutton(enhance,text="Average Blur 7x7",activebackground="#004400",variable=x,value=3,indicatoron = 0,width = 15,padx = 15,command=partial(smooth1,img,enhance,x))
    ls9=tk.Radiobutton(enhance,text="Median Blur",activebackground="#004400",variable=x,value=4,indicatoron = 0,width = 15,padx = 15,command=partial(smooth1,img,enhance,x))
    ls10=tk.Radiobutton(enhance,text="Gaussian Blur",activebackground="#004400",variable=x,value=5,indicatoron = 0,width = 15,padx = 15,command=partial(smooth1,img,enhance,x))
    ls6.place(x=900,y=250)
    ls7.place(x=900,y=280)
    ls8.place(x=900,y=310)
    ls9.place(x=900,y=340)
    ls10.place(x=900,y=370)
    
    ls11=tk.Radiobutton(enhance,text="Image Negative",activebackground="#004400",variable=x,value=1,indicatoron = 0,width = 15,padx = 15,command=partial(point1,img,enhance,x))
    ls12=tk.Radiobutton(enhance,text="Contrast",activebackground="#004400",variable=x,value=2,indicatoron = 0,width = 15,padx = 15,command=partial(cont,img,y1,enhance)) 
    ls13=tk.Radiobutton(enhance,text="Log Transform",activebackground="#004400",variable=x,value=3,indicatoron = 0,width = 15,padx = 15,command=partial(point1,img,enhance,x))
    ls14=tk.Radiobutton(enhance,text="Power Law",activebackground="#004400",variable=x,value=4,indicatoron = 0,width = 15,padx = 15,command=partial(power,text,img,enhance))
    ls15=tk.Radiobutton(enhance,text="Binary",activebackground="#004400",variable=x,value=5,indicatoron = 0,width = 15,padx = 15,command=partial(point1,img,enhance,x))
    ls16=tk.Radiobutton(enhance,text="Brightness",activebackground="#004400",variable=x,value=6,indicatoron = 0,width = 15,padx = 15,command=partial(brightness,z,img,enhance))
    ls11.place(x=1050,y=250)
    ls12.place(x=1050,y=280)
    ls13.place(x=1050,y=310)
    ls14.place(x=1050,y=340)
    ls15.place(x=1050,y=370)
    ls16.place(x=1050,y=400)
    
    ls21=tk.Radiobutton(enhance,text="Fourier filter",activebackground="#004400",variable=x,value=1,indicatoron = 0,width = 15,padx = 15,command=partial(fourier1,img,enhance))
    ls21.place(x=550,y=250)
    
    enhance.mainloop()
    
#for main window
def submit():
    window.geometry("1000x600")
    text=entrytext.get()
    img=cv2.imread(text)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dim=(400,400)
    res=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    im = Image.fromarray(res)
    imgtk = ImageTk.PhotoImage(image=im) 
    imglabel=tk.Label(window,image=imgtk)
    imglabel.image=imgtk
    entrytext.place(x=550,y=150)
    imglabel.place(x=100,y=150)
    bt.place(x=550,y=200)
    bt1.place(x=650,y=200)
    imgrestore=tk.Button(window,text="Image Restoration",font=("Arial Bold",15),bg='#ff7744',fg='#005500',activebackground="#ff6644",command=partial(restoration,res))
    imgenhance=tk.Button(window,text="Image Enhancement",font=("Arial Bold",15),bg='#ff7744',fg='#005500',activebackground="#ff6644",command=partial(enhancement,res))
    imgrestore.place(x=550,y=250)
    imgenhance.place(x=550,y=300)

        
window=tk.Tk()
var=tk.StringVar()
window.title("GUI")
window.configure(bg='#ff7744')
window.geometry('500x300')
label=tk.Label(window,text="Image Manipulation",font=("Arial Bold",20),fg="blue",bg='#ff7744')
lb=tk.Label(window,text="Give the full path of your image:",font=("Arial Bold",15),bg='#ff7744')
entrytext=tk.Entry(window,bd=5,textvariable=var,width=50)
bt=tk.Button(window,text="SUBMIT",activebackground='#ff8844',font=("Arial Bold",15),fg='black',command=submit)
bt1=tk.Button(window,text="EXIT",activebackground='#ff8844',font=("Arial Bold",15),fg='black',command=partial(back,window))
label.place(x=125,y=50)
lb.place(x=80,y=100)
entrytext.place(x=100,y=150)
bt.place(x=150,y=200)
bt1.place(x=250,y=200)
window.mainloop()
