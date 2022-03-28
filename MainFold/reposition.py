# ycl7199
# use char coordinates to calculate precise string bounding box 
import csv 
import json
import os
import cv2
import numpy as np
from shapely.geometry import Polygon

def get_iou_partial(g, p):
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = p.area
    if union == 0:
        return 0
    else:
        return inter/union

def get_new_bbox_horizontal(char_list):
    x1_list = [i[0] for i in char_list]
    x3_list = [i[4] for i in char_list]
    x1_min_index = x1_list.index(min(x1_list))
    x3_max_index = x3_list.index(max(x3_list))
    new_bbox = [char_list[x1_min_index][0], char_list[x1_min_index][1], char_list[x1_min_index][2], char_list[x1_min_index][3],
                char_list[x3_max_index][4], char_list[x3_max_index][5], char_list[x3_max_index][6], char_list[x3_max_index][7]]
    return new_bbox

def get_new_bbox_vertical(char_list):
    y1_list = [i[1] for i in char_list]
    y3_list = [i[3] for i in char_list]
    y1_min_index = y1_list.index(min(y1_list))
    y3_max_index = y3_list.index(max(y3_list))
    new_bbox = [
        char_list[y1_min_index][0], char_list[y1_min_index][1], char_list[y3_max_index][2], char_list[y3_max_index][3],
        char_list[y3_max_index][4], char_list[y3_max_index][5],  char_list[y1_min_index][6], char_list[y1_min_index][7]
    ]
    return new_bbox



### open json file to do reposition
with open('json_out3_3.json', 'r') as fp:
    public_json = json.load(fp)

for i, img in enumerate(public_json):
    for str_ in public_json[img]:
        
        char_list = public_json[img][str_]['char_coor']
        str_coor = [float(i) for i in public_json[img][str_]['str_coor'][0]]
        str_height =str_coor[3] - str_coor[1]
        str_width = str_coor[4] - str_coor[0]
        if str_height > str_width:
            new_str_coor = get_new_bbox_vertical(char_list)
            new_w = [new_str_coor[6]-new_str_coor[0], new_str_coor[4]-new_str_coor[2]]
            if max(new_w)/min(new_w) > 1.4:
                new_str_coor = str_coor

        else:
            new_str_coor = get_new_bbox_horizontal(char_list)
            new_h = [new_str_coor[3]-new_str_coor[1], new_str_coor[5]-new_str_coor[7]]
            if max(new_h)/min(new_h) > 1.4:
                new_str_coor = str_coor
        public_json[img][str_]['new_str_coor'] = new_str_coor
        new_char_coor = []
        for char in char_list: ### calculate char&new str iou to reduce incorrect char coor
            iou = get_iou_partial(new_str_coor, char)
            if iou > 0.6:
                new_char_coor.append(char)
        public_json[img][str_]['new_char_coor'] = new_char_coor

        

### open new json file to save result
with open('json_out3_adj_char.json', 'w') as fp:
    json.dump(public_json, fp)
        
    
