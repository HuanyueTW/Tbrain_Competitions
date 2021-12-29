# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:45:10 2021

@author: brian
"""

import os
import json
import cv2
import matplotlib.pyplot as plt

all_ = 0
p = 0
#with open("json_continue_zh_en_mix_.json", encoding="utf-8") as f:
with open("pri_json_3_adj.json", encoding="utf-8") as f:  
    data = json.load(f)
    
    # # 輸出縮排Json
    # jsonData_sort = json.dumps(data, sort_keys = True, indent=4)
    # print(jsonData_sort)
    
    for i in data: # 照片
        img = cv2.imread("p/" + i + ".jpg")
        
        for j in data[i]: # 圖片內題目
            p += 1
            
            xy = data[i][j]["new_str_coor"]
            

            yrange = int(float(xy[5])) - int(float(xy[1]))
            xrange = int(float(xy[4])) - int(float(xy[0]))
            
            xr = True
            if(yrange > xrange):
                xr = False
            
            
        
            for k in data[i][j]["new_char_coor"]:
                k = [ int(i) for i in k ]

                all_ += 1

                if k[1] < 0:
                    k[1] = 0
                if k[5] < 0:
                    k[5] = 0
                if k[0] < 0:
                    k[0] = 0
                if k[4] < 0:
                    k[4] = 0
               
                #print(i, k[1], k[5], k[0], k[4])
                #cv2.imwrite("out/" + i + "_" + str(p).zfill(4) + "_" + str(all_).zfill(5) + ".jpg", img[k[1] : k[5], k[0] : k[4]])
                if(xr):
                    cv2.imwrite("outF/"+ str(p).zfill(5) + "_" + str(k[0]).zfill(5) + ".jpg", img[k[1] : k[5], k[0] : k[4]])
                else:
                    cv2.imwrite("outF/"+ str(p).zfill(5) + "_" + str(k[1]).zfill(5) + ".jpg", img[k[1] : k[5], k[0] : k[4]])

            
    
    # # li = list(data.items())
    # # print(list(li[0][1]))
    # # img_li = list(li[0][1])
    
    # target_img = list(data)
    # print(target_img)
    # target_string = list(data[target_img[0]])
    # print(target_string)
    
    # print(data[target_img[0]])
    
    
    