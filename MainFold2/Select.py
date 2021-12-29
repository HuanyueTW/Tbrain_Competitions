# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:24:21 2021

@author: brian
"""

import os
import shutil
import random

path = "Class2/"
percent = 0.9 #train data
os.mkdir("done/train")
os.mkdir("done/valid")
os.mkdir("done/test")

for i in os.listdir(path):
    num = len(os.listdir(path + i))
    select_list = random.sample(range(0, num), round(num * percent))
    fold = os.listdir(path + "/" + i)
    os.mkdir("done/train/" + i)
    os.mkdir("done/test/" + i)
    
    for data in range(0, num):
        if data in select_list:        
            shutil.copy("Class2/" + i + "/" + fold[data], "done/train/" + i + "/" + fold[data])
            
        else:  
            shutil.copy("Class2/" + i + "/" + fold[data], "done/test/" + i + "/" + fold[data])

path = "done/train/"
for i in os.listdir(path):
    num = len(os.listdir(path + i))
    select_list = random.sample(range(0, num), round(num * 0.8))
    fold = os.listdir(path + "/" + i)
    os.mkdir("done/valid/" + i)
    
    for data in range(0, num):
        if data not in select_list:         
            shutil.move(path + i + "/" + fold[data], "done/valid/" + i + "/" + fold[data])
             