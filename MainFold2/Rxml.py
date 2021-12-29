# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:26:07 2021

@author: brian
"""

import xml.etree.ElementTree as ET
import os 
import cv2

for i in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']:
    os.mkdir("Class/" + i + "-")
    
for i in os.listdir("xml_all"):
    
    img = cv2.imread("add/" + i[:-4] + ".png")    

    # 從檔案載入並解析 XML 資料
    tree = ET.parse('xml_all/' + i)
    root = tree.getroot()

    name = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    
    for neighbor in root.iter('name'):
        name.append(neighbor.text)
    for neighbor in root.iter('xmin'):
        xmin.append(int(neighbor.text))
    for neighbor in root.iter('xmax'):
        xmax.append(int(neighbor.text))
    for neighbor in root.iter('ymin'):
        ymin.append(int(neighbor.text))
    for neighbor in root.iter('ymax'):
        ymax.append(int(neighbor.text))
    
    for i in range(len(name)):
        
        if name[i] in ["word", "zh"]:
            continue
        
        o_img = img[ymin[i]:ymax[i], xmin[i]:xmax[i]]
        if name[i] in  ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] :
            cv2.imwrite("Class/" + name[i] + "-/" + str(name[i]) + "_" + str(xmin[i]) + "_" + str(xmax[i]) + ".png",
                    o_img)
        else:
            cv2.imwrite("Class/" + name[i] + "/" + str(name[i]) + "_" + str(xmin[i]) + "_" + str(xmax[i]) + ".png",
                    o_img)
    