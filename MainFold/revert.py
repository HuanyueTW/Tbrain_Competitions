# ycl7199
# convert char coordinates from string image to original image
import csv 
import json
import os
import cv2
import numpy as np


### open exist json to continue or create a new dict to start 
json_continue = {}
# with open('json_output.json', 'r') as fp:
#     json_continue = json.load(fp)


### open json file crate by padding.py to load string coordinates in padding image
with open('bbox_info_mix.json', 'r') as fp:
    padding_value = json.load(fp)
print(padding_value)
origin_info = {}

### list char detection result(csv file) in input folder
csv_list = os.listdir('char_csv_mix_out3')
print(csv_list)


for per_csv in csv_list: # generate origin csv's dict
    file_name = per_csv.split('_')[0] + '_' + per_csv.split('_')[1]
    file_path = 'str_csv\\' + file_name + '.csv'
    if file_name not in origin_info:
        origin_info[file_name] = {}
        print(file_name)
        with open(file_path, 'r', encoding= 'utf-8-sig') as data: # per 
            csv_reader = csv.reader(data)
            for str_data in csv_reader:
                # print(str_data)
                str_coor = str_data[2:10]
                str_coor = [float(i) for i in str_coor]
                origin_info[file_name][str_data[0]] = []
                origin_info[file_name][str_data[0]].append(str_coor)
                origin_info[file_name][str_data[0]].append(str_data[1])
             
                
for csv_f in csv_list: # iterate every string(csv)
    file_path = 'char_csv_mix_out3\\' + csv_f
    image_name = csv_f.split('_')[0] + '_' + csv_f.split('_')[1]
    str_name = csv_f.split('.')[0]
    # print(image_name)
    if image_name not in json_continue:
        json_continue[image_name] = {}
    json_continue[image_name][str_name] = {}
    json_continue[image_name][str_name]['str_coor'] = origin_info[image_name][str_name]
    json_continue[image_name][str_name]['char_coor'] = []
    with open(file_path, 'r', encoding= 'utf-8-sig') as data:
        csv_reader = csv.reader(data)
        
        # print(str_name)
        for i in csv_reader: # iterate every char
            # str_name = i[0].split('_')[:-1]
            # str_name = str_name[0] + '_' + str_name[1] + '_' + str_name[2]
            # print(str_name)
            # print(padding_value[str_name])
            char_coor = i[2:10]
            char_coor = [float(i) for i in char_coor]
            if ((char_coor[4]) - char_coor[0]) * (char_coor[5] - char_coor[1]) >= 200:
                for j,coor in enumerate(char_coor): # iterate char coordinates & relocation
                    # print(float(coor))
                    # print(float(padding_value[str_name][0]))
                    coor_new = float(char_coor[j])
                    if j%2 == 0:
                        padding = float(padding_value[str_name][0])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][0])
                        char_coor[j] = coor_new - padding + origin_coor - 8
                        
                    else:
                        padding = float(padding_value[str_name][1])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][1])
                        char_coor[j] = coor_new - padding + origin_coor -8
            else:
                for j,coor in enumerate(char_coor): # iterate char coordinates & relocation
                    # print(float(coor))
                    # print(float(padding_value[str_name][0]))
                    coor_new = float(char_coor[j])
                    if j%2 == 0:
                        padding = float(padding_value[str_name][0])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][0])
                        char_coor[j] = coor_new - padding + origin_coor -4
                        
                    else:
                        padding = float(padding_value[str_name][1])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][1])
                        char_coor[j] = coor_new - padding + origin_coor -4
                
            # print(char_coor)
            json_continue[image_name][str_name]['char_coor'].append(char_coor)



### save result in json file
with open('json_output.json', 'w') as fp:
    json.dump(json_continue, fp)


