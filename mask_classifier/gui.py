# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:12:56 2020

@author: Bima-aiforesee
"""

from tkinter import *
import os

root = Tk(className = 'face_recognition_gui')
root.title('Mask and Termal Detector')

#==== Tittle
l = Label(root, text="Mask and Termal Detector")
l.config(font=("Courier", 20))
l.pack()

f=Frame(root,height=1, width=400, bg="black")
f.pack()

#===== Recognize Button

l = Label(root, text="Sistem Start")
l.config(font=("Courier", 15))
l.pack()

def recog_lbph_btn_load():
    os.system('python fix_play.py')

recogL_btn = Button(root,text="Ok", command=recog_lbph_btn_load)
recogL_btn.pack()

root.mainloop()
