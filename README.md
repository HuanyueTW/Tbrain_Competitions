# 繁體中文場景文字辨識-高階賽 程式碼說明

## 組別：team_52
## 成員：蔣明憲 劉益誠

## 目錄
- <a href="#dependencies" > 環境套件</a>
- <a href="#installation" > 安裝方式</a>
- <a href="#fold" > 資料夾布局</a>
- <a href="#Detect" > 偵測 (按流程排列)</a>
    - <a href="#preprocess" > 前處理</a>
        - <a href="#JsonforUse" > json2yolo.py ： 將官方提供的 json 格式 label 傳換為 yolo 用的 txt 格式 label</a>
        - <a href="#VocforUse" > voc2yolo.py ： 將自己製作的 xml 格式 label 傳換為 yolo 用的 txt 格式 label</a>
        - <a href="#JsonforUse" > dataset_split.py ： 進行訓練資料集分割 (train, validation, test)</a>
    - <a href="#DetectModelTraining" > 模型訓練 (字元與字串)</a>
        - <a href="#YOLOv5" > YOLOv5</a>
    - <a href="#Detection_str" > 模型偵測1</a>
        - <a href="#RunningDetect_str" > detect_csv.py ： 進行字串偵測</a>
    - <a href="#Midprocess" > 中間處理</a>  
        - <a href="#Padding" > padding.py ： 將字串偵測後裁切下來的字串圖進行補底以執行字元偵測</a>
    - <a href="#Detection_char" > 模型偵測2</a>  
        - <a href="#RunningDetect_char" > detect_csv.py ： 進行字元偵測</a>
    - <a href="#afterProcessing" > 偵測後處理</a>  
         - <a href="#WriteJson" > revert.py ： 讀取字元字串偵測結果與先前存下的座標紀錄檔來進行座標計算與結果合併</a>
        - <a href="#reposition" > reposition.py ： 進行字串座標框的調整以及字元座標組的IOU計算與調整</a>
- <a href="#Classify" > 分類</a>
    - <a href="#preprocess2" > 前處理-偵測答案處理</a>
        - <a href="#ReadJson" > ReadJson.py ： 利用偵測製作出的 json，將字元題目切割出來</a>
        - <a href="#WriteCsv" > WriteCsv.py ： 利用偵測製作出的 json，根據一個字串一題的原則做成座標csv</a>
        - <a href="#revise" > revise.py ： 將座標 csv 欄位順序轉成比賽答案格式</a>
    - <a href="#preprocess2-1" > 前處理-分類訓練資料處理</a>
        - <a href="#part" > part.py ： 將官方 traindata 的中文單字提取出來</a>
        - <a href="#find_numletter_and_writexml" > find_numletter_and_writexml.py ： 手動標記英文數字的public data</a>
        - <a href="#Rxml" > Rxml.py ： 手動標記英文數字的public data</a>
        - <a href="#Class" > Class.py ： 從前步驟做好的做好的資料夾進行類別處理</a> 
        - <a href="#Select" > Select.py ： 從前步驟做好的英文數字標記檔提取英文數字樣本</a> 
    - <a href="#Train" > 模型訓練</a>
        - <a href="#ModelTrain" > InceptionResNetV2.py ： 訓練分類模型InceptionResNetV2</a> 
    - <a href="#Main" > 主程式</a>
        - <a href="#Classification" > Classification.py ： 分類主程式</a>
    - <a href="#afterprocess2" > 後處理</a>
        - <a href="#add_ans" > add_ans.py ： 將座標 csv 與分類答案 csv 合併</a>    



<div id="dependencies"></div>

## 環境套件


以下是我們組在運行程式時所使用的環境套件與套件： 
### 偵測主機 (有Cuda、Cudnn即可加速)
 - Python 3.8.1
 - torch 1.7.1
 - torchvision>=0.8.1
 - matplotlib>=3.2.2
 - numpy>=1.18.5
 - opencv-python>=4.1.2
 - Pillow>=7.1.2
 - PyYAML>=5.3.1
 - requests>=2.23.0
 - scipy>=1.4.1
 - tqdm>=4.41.0
 - tensorboard>=2.4.1
 - pandas>=1.1.4
 - seaborn>=0.11.0


### 分類主機
 - Python 3.8.8 (Anaconda 2021.05)
 - Tensorflow 2.3.0
 - Keras 2.4.3 
 - Opencv-python 4.5.3.56
 - Numpy 1.18.5
 - json 0.9.5
 - shutil 

 分類如果有加速需求才需要下載這個：

- Tensorflow-gpu 2.3.0


<div id="installation"></div>

## 安裝方式(其他套件亦相同)

**Tensorflow** 或者 **Tensorflow GPU** , 需搭配 CUDA(10.1) 及 cuDNN(7.6.5) 安裝才可使用 GPU 加速
```bash
pip install tensorflow==2.3.0
pip install tensorflow-gpu==2.3.0
```
**ImageAI**
```bash
pip install imageai
```

<div id="fold"></div>

## 資料夾布局(最終)
### 偵測主機
```
>> MainFold     >> json2yolo.py
                >> voc2yolo.py
                >> dataset_split.py
                >> padding.py
                >> revert.py
                >> reposition.py
                >> detect_csv.py


                >> img (train 解壓縮)
                >> json (train 解壓縮)
                >> img_public (public 解壓縮)
                >> annotations_str
                >> label
                >> yolo_labels
                >> txt          
                >> images       >> train
                                >> valid
                                >> test
                >> labels       >> train
                                >> valid
                                >> test
```
### 分類主機
```
>> MainFold2    >> ReadJson.py
                >> WriteCsv.py
                >> revise.py
                >> part.py
                >> find_numletter_and_writexml.py
                >> Rxml.py
                >> Class.py
                >> Select.py
                >> InceptionResNetV2.py
                >> Classification.py
                >> add_ans.py


                >> img (train 解壓縮)
                >> json (train 解壓縮)
                >> public (public 解壓縮)
                >> part_img
                >> xml_all
                >> Class
                >> done       >> train
                              >> valid
                              >> test

```


<div id="Detect"></div>

## -----------------偵測-----------------

<div id="preprocess"></div>

## 前處理

<div id="JsonforUse"></div>

### 前處理-將官方提供的 json 格式 label 傳換為 yolo 用的 txt 格式 label

### json2yolo.py
#### **將json標註檔轉換成txt標註檔**
&ensp;&ensp;&ensp;&ensp;首先準備一個 json 資料夾放置從 train 資料集取出的 json 檔案以及準備 annotations_str 資料夾來放置輸出的 txt 檔案，此程式是要製作字串 (groupid = 0, 2, 3) 以及中文字元 (groupid = 1) 的兩個偵測模型。
```python
import json
import os

def coordinate_convert(img_w, img_h, x_min, x_max, y_min, y_max):
    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2
    width = x_max - x_min
    height = y_max - y_min
    center_x /= img_w
    center_y /= img_h
    width /= img_w
    height /= img_h
    return center_x, center_y, width, height


def json2yolo(json_filename):
    file_path = '.\\json\\' + json_filename
    with open(file_path, encoding="utf-8-sig") as f:
        content = json.load(f)
        # print(content['shapes'])
        img_w = float(content['imageWidth'])
        img_h = float(content['imageHeight'])
        str_ch = []
        char_ch = []
        str_en = []
        mix = []
        single_ch = []
        
        for bbox in content['shapes']:
            # print(bbox)
            x_min = min([i[0] for i in bbox['points']])
            x_max = max([i[0] for i in bbox['points']])
            y_min = min([i[1] for i in bbox['points']])
            y_max = max([i[1] for i in bbox['points']])

            if bbox['group_id'] == 0:
                c_x, c_y, width, height = coordinate_convert(img_w, img_h, x_min, x_max, y_min, y_max)
                str_ch.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(0, c_x, c_y, width, height))
            elif bbox['group_id'] == 2:
                c_x, c_y, width, height = coordinate_convert(img_w, img_h, x_min, x_max, y_min, y_max)
                str_en.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(1, c_x, c_y, width, height))
            elif bbox['group_id'] == 3:
                c_x, c_y, width, height = coordinate_convert(img_w, img_h, x_min, x_max, y_min, y_max)
                mix.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(2, c_x, c_y, width, height))


        all_str = str_ch + str_en + mix
        ### save yolo format txt file to annotations_str folder
        save_file_name_all = os.path.join("annotations_str", json_filename.replace("json", "txt"))   
        print("\n".join(all_str), file= open(save_file_name_all, "w"))


    # print('process done')      
                
json_folder = 'json'
file_list = os.listdir(json_folder)
print(file_list)
for i, filename in enumerate(file_list):
    json2yolo(filename)
    print('{}/{}'.format(i, len(file_list)))
```

<div id="VocforUse"></div>

### 前處理-將自己製作的 xml(Labelimg make) 格式 label 傳換為 yolo 用的 txt 格式 label

### voc2yolo.py
#### **將voc標註檔轉換成txt標註檔**
準備 label 資料夾放置用 labelimg 做出 voc 格式的 xml 標註檔，另外準備 yolo_labels 放置輸出的 txt 標註檔，此程式則是因為英數跟混合字串官方沒有提供，只能自己用 labelimg 製作出 xml 檔案，因此在訓練英數字元偵測或混合字元偵測兩個模型前需要先製作好標註檔。
```python
from os import name
from typing import Text
import xml.etree.ElementTree as ET
import os

def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    info_dict = {}
    info_dict['bboxes'] = []

    for elem in root:
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))          
            info_dict['image_size'] = tuple(image_size)
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)  
    return info_dict



def convert_to_yolov5(info_dict):
    print_buffer = []
    for b in info_dict["bboxes"]:
        if b['class'] == 'zh' or 'word': ### 挑選出中文字元的label，如果要準備混和字元偵測的訓練標籤則需要將class_id的註解取消，並將continue註解掉。英數字元偵測則不需修改。
            # class_id = 1
            continue
        elif b['class'].isdigit(): 
            class_id = 0
        else:
            class_id = 0            
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    save_file_name = os.path.join("yolo_labels", info_dict["filename"].replace("jpg", "txt"))

    print("\n".join(print_buffer), file= open(save_file_name, "a"))        

file_list = os.listdir('labels')
print(file_list)
for file_name in file_list:
    info_dict = extract_info_from_xml('labels/' + file_name)
    convert_to_yolov5(info_dict)


```

### 前處理-進行訓練資料集分割 (train,validation,test)

<div id="datasetProcess"></div>

### dataset_split.py

#### **將資料集分割為train,test,validation部分來進行訓練 (字元與字串共用做法)**
&ensp;&ensp;&ensp;&ensp;先創建 img 與 txt 資料夾，將要進行分割的資料集影像與標註檔案分別放置到 img 與 txt 資料夾。另外準備 images 與 labels 資料夾且兩個資料夾內都先準備好 test, val, train 三個資料夾來放置照設定比例隨機分割後的影像與標註檔。
```python

import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

images = [os.path.join('img', x) for x in os.listdir('img')]
annotations = [os.path.join('txt', x) for x in os.listdir('txt') if x[-3:] == "txt"]

images.sort()
annotations.sort()
print(images)
print(annotations)

train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

print(train_images, val_images, train_annotations, val_annotations)

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        print(f)
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'labels/train/')
move_files_to_folder(val_annotations, 'labels/val/')
move_files_to_folder(test_annotations, 'labels/test/')
```



<div id="DetectModelTraining"></div>

## 偵測模型訓練
&ensp;&ensp;&ensp;&ensp;要注意的是，這裡的偵測訓練由於會需要訓練四個模型，但訓練都是同樣 cmd 語法，稍微有點混亂，因此建議一次一個任務來做，譬如先針對字串的訓練檔案轉 txt 並且切割好後，跑完訓練，再重置剛剛的資料夾換去做中文字元，依此類推再做英數字串跟混合字串的偵測模型。

<div id="YOLOv5"></div>

### **YOLOv5 (字元與字串共用做法) **
&ensp;&ensp;&ensp;&ensp;先從 yolov5 的官網 (https://github.com/ultralytics/yolov5) 上將 yolov5 的專案下載，或是從附檔內的 yolov5 專案壓縮檔解壓縮。
接著修改 (或新增) Tbrain.yaml 來更改訓練資料集的參數。  
&ensp;&ensp;&ensp;&ensp;將訓練資料集的 train, val, test 資料夾路徑修改至 Tbrain.yaml，接著修改 class 數量(例如進行三種字串偵測 class 數量就是3)，接著用以下範例 command 來進行訓練。
(command 說明: --cfg 參數為欲使用的模型架構檔案路徑， --batch 為訓練時的 batch size， --epochs 為訓練回合數， --data 為資料集資料集yaml檔案路徑， --weights 為欲使用的訓練權重 (使用官方提供的 yolov5m pretrain weight), --name為訓練模型資料夾名稱 (程式會自動在 run 資料夾內建立相對應的資料夾儲存訓練結果)。
```
python train.py --img 640 --cfg yolov5m.yaml  --batch 8 --epochs 60 --data Tbrain.yaml --weights yolov5m.pt  --name yolov5m_Tbrain_str
```

<div id="Detection_str"></div>

## 模型偵測1
&ensp;&ensp;&ensp;&ensp;要注意的是，這裡的偵測由於會需要四個模型權重，但使用同樣 cmd 語法，稍微有點混亂，因此建議一次一個任務來做，且這裡要注意順序是先做好字串偵測後，得出三種類型的物件，再去針對性的用三個偵測字元模型去做，就會得出要給分類模型的字元座標。

<div id="RunningDetect_str"></div>

###  detect_csv.py

#### **進行字串偵測 (字元與字串共用做法)**
&ensp;&ensp;&ensp;&ensp;將附檔內的 detect_csv.py 複製到 yolov5 專案資料夾， detect_csv.py 將原本 detect.py 的 --save-txt 功能改為輸出 csv 格式的 label 以對應比賽需求。  
&ensp;&ensp;&ensp;&ensp;執行以下 command 進行偵測任務，參數說明: --source 欲進行偵測的圖片資料夾位置，--weight 要使用的模型檔案位置， --conf 偵測時的 confidence 閥值設定， --name 偵測結果的資料夾名稱(跟訓練時一樣程式會自動在 run 資料夾內建立相對應的資料夾來儲存結果)，--save-txt 儲存 bounding box 座標， --save-crop 將物件框從圖片中裁切下來另存成新的圖片。最後會得出一個偵測字串的Csv資料夾，以及三種字元偵測各自的csv資料夾。

```
python detect_csv.py --source test --weights runs/train/yolov5m_Tbrain_char2/weights/best.pt --conf 0.25 --name yolov5m_Tbrain_str2 --save-txt --save-crop
```

<div id="Midprocess"></div>

## 中間處理

<div id="Padding"></div>

### padding.py
#### **將字串偵測後裁切下來的字串圖進行補底以執行字元偵測**
&ensp;&ensp;&ensp;&ensp;將要進行補底的字串圖片資料夾放在與 padding.py 相同目錄下，並修改好 folder_name 為對應資料夾名稱，程式另外存下的 json 檔為後續處理將字座標還原至對應原圖時會用到的座標紀錄檔。
```python
import cv2
import numpy as np
import os
import json

def padding(filename, img_size ,bbox_info):
    ### read image
    img = cv2.imread(filename)
    old_image_height, old_image_width, channels = img.shape

    new_image_width = img_size
    new_image_height = img_size
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    ### compute center to put image
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2


    ### copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    
    print('x center = {}, y center = {}'.format(x_center,y_center))
    b_name = filename.split('.')[0].split('\\')[1]
    ### record center point
    bbox_info[b_name] = [x_center, y_center]
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print( filename.split('\\')[-1])
    save_name = filename.split('\\')[-1]
    save_path = os.path.join(folder_name +'_border\\',save_name)
    ### save result
    cv2.imwrite(save_path, result)

### Input folder
folder_name = 'str_mix'
file_list = os.listdir(folder_name)
print(file_list)
bbox_info = {}
for filename in file_list:
    file_path = folder_name +'\\' + filename
    try:
        padding(file_path, 640, bbox_info)
    except:
        try:
            padding(file_path, 1280, bbox_info)
        except:
            try:
                padding(file_path, 2160, bbox_info)
            except:
                padding(file_path, 4096, bbox_info)
print(bbox_info)

### save center point record
with open('bbox_info_mix.json', 'w') as fp:
    json.dump(bbox_info, fp)
```


<div id="Detection_char"></div>

## 模型偵測2

<div id="RunningDetect_char"></div>

###  detect_csv.py

#### **進行字元偵測 (字元與字串共用做法)**
&ensp;&ensp;&ensp;&ensp;同字串偵測，只需指定到之前訓練好的字元模型以及字串偵測出且補底好的圖片集即可。
```
python detect_csv.py --source test --weights runs/train/yolov5m_Tbrain_char2/weights/best.pt --conf 0.25 --name yolov5m_Tbrain_str2 --save-txt --save-crop
```



<div id="afterProcessing"></div>

## 偵測後處理

<div id="WriteJson"></div>

### 後處理-將前面字串與字元偵測後的座標csv檔案進行調整與合併成json輸出

### revert.py
#### **讀取字元字串偵測結果與先前存下的座標紀錄檔來進行座標計算與結果合併**

&ensp;&ensp;&ensp;&ensp;先修改好欲讀取的座標紀錄檔名稱，接著將字元偵測結果 csv 資料夾(例如:char_csv_mix_out3)與字串偵測結果 csv 資料夾(例如:str_csv)放到與程式相同的目錄下並修改相對應參數。程式執行完畢後會輸出一個 json 檔案儲存所有結果，由於有三個字元偵測的 CSV 資料夾，因此此程式需要做三次，將三個字元結果與字串結果合併。
```python
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
            char_coor = i[2:10]
            char_coor = [float(i) for i in char_coor]
            if ((char_coor[4]) - char_coor[0]) * (char_coor[5] - char_coor[1]) >= 200:
                for j,coor in enumerate(char_coor): # iterate char coordinates & relocation
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
                    coor_new = float(char_coor[j])
                    if j%2 == 0:
                        padding = float(padding_value[str_name][0])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][0])
                        char_coor[j] = coor_new - padding + origin_coor -4
                        
                    else:
                        padding = float(padding_value[str_name][1])
                        origin_coor = float(json_continue[image_name][str_name]['str_coor'][0][1])
                        char_coor[j] = coor_new - padding + origin_coor -4
        
            json_continue[image_name][str_name]['char_coor'].append(char_coor)



### save result in json file
with open('json_output.json', 'w') as fp:
    json.dump(json_continue, fp)
```

### 後處理-reposition

<div id="reposition"></div>

### reposition.py
#### **進行字串座標框的調整以及字元座標組的IOU計算與調整**
&ensp;&ensp;&ensp;&ensp;讀取 revert.py 產生的 json 檔案(例如:json_out.json)，利用裡面的字元座標組來進行字串座標的 reposition 以及字元座標組的 IOU 計算與調整，最後輸出的 json 檔(json_out_reposition.json) 則會是原本讀取的資訊加上計算後的結果資訊合併出的結果，之後可以使用進行偵測的原圖以及此 json 檔來進行字元分類。

```python
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
with open('json_out.json', 'r') as fp:
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
with open('json_out_reposition.json', 'w') as fp:
    json.dump(public_json, fp)
```




<div id="Classify"></div>

## -----------------分類-----------------

<div id="preprocess2"></div>

## 前處理-偵測答案處理


<div id="ReadJson"></div>

### **前處理-利用偵測製作出的 json，將字元題目切割出來**
### ReadJson.py
&ensp;&ensp;&ensp;&ensp;由於為了降低接下來的分類負擔，事先將字元從圖片上切下來做成題目集，也方便之後對圖片做處理。
```python
# -*- coding: utf-8 -*-
import os
import json
import cv2
import matplotlib.pyplot as plt

all_ = 0
p = 0
FileName = "pri_json_3_adj.json"
#with open("json_continue_zh_en_mix_.json", encoding="utf-8") as f:
with open(FileName, encoding="utf-8") as f:  
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
            
            # 垂直水平判斷
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
               
                #根據垂直水平判斷來決定存圖的名稱間接影響分類順序
                if(xr):
                    cv2.imwrite("outF/"+ str(p).zfill(5) + "_" + str(k[0]).zfill(5) + ".jpg", img[k[1] : k[5], k[0] : k[4]])
                else:
                    cv2.imwrite("outF/"+ str(p).zfill(5) + "_" + str(k[1]).zfill(5) + ".jpg", img[k[1] : k[5], k[0] : k[4]])

```

<div id="WriteCsv"></div>

### **前處理-利用偵測製作出的 json，根據一個字串一題的原則做成座標csv**
### WriteCsv.py
&ensp;&ensp;&ensp;&ensp;將偵測最後結果json檔做提取題目的部分，每一題代表一個字串，此舉也是為符合繳交答案格式。
```python
# -*- coding: utf-8 -*-
import json
import pandas as pd

with open("pri_json_3_adj.json", encoding="utf-8") as f:  
    data = json.load(f)
    
    all_ans = []
    for i in data: # 照片
        
        for j in data[i]: # 圖片內題目
            ans = []
            ans.append(i)
            for k in data[i][j]["new_str_coor"]:
            #for l in data[i][j]["new_str_coor"]:
                ans.append(int(float(k)))
            all_ans.append(ans)

#out_ans = pd.read_csv("output.csv")

new_df = pd.DataFrame(all_ans)
#new_df = pd.concat([new_df, out_ans], axis = 1)
#pd.DataFrame.to_csv(new_df, "submission.csv", header=False, index=False)

pd.DataFrame.to_csv(new_df, "submission_pri.csv", header=False, index=False)

```

<div id="revise"></div>

### **前處理-將座標csv欄位順序轉成比賽答案格式**
### revise.py
&ensp;&ensp;&ensp;&ensp;前面偵測出的座標並未依照答案格式逆時針順序，因此需要進行改變欄位順序的處理。
```python
# -*- coding: utf-8 -*-
import pandas as pd

#分類答案csv
out_ans = pd.read_csv("submission_v4_2.csv", header = None)
#偵測字串座標csv
# = pd.read_csv("submission_v4_2.csv", header = None)

out_ans = pd.DataFrame(out_ans)

a=out_ans[1]
b=out_ans[2]
c=pd.concat([a, b], axis = 1)

a=out_ans[3]
b=out_ans[4]
d=pd.concat([a, b], axis = 1)

a=out_ans[5]
b=out_ans[6]
e=pd.concat([a, b], axis = 1)

a=out_ans[7]
b=out_ans[8]
f=pd.concat([a, b], axis = 1)

out_ans = out_ans.drop(1, axis = 1)
out_ans = out_ans.drop(2, axis = 1)
out_ans = out_ans.drop(3, axis = 1)
out_ans = out_ans.drop(4, axis = 1)
out_ans = out_ans.drop(5, axis = 1)
out_ans = out_ans.drop(6, axis = 1)
out_ans = out_ans.drop(7, axis = 1)
out_ans = out_ans.drop(8, axis = 1)

out_ans=pd.concat([out_ans, c, f, e, d], axis = 1)

print(out_ans)

pd.DataFrame.to_csv(out_ans, "submission_v4_2.csv", header=False, index=False)

```

<div id="preprocess2-1"></div>

## 前處理-分類訓練資料處理 

<div id="part"></div>

### **前處理-將官方 train data 的中文單字提取出來**
### part.py
&ensp;&ensp;&ensp;&ensp;只要將該程式跟 train 資料解壓縮後的檔案放一起，並且新增資料夾 part_img 來放置，其中程式內的group_id 等於 1 跟 4 是我們看過照片後選擇出來的，4 的部分看需求來決定要不要放進去一起訓練，完成後會將切出的圖片放到 part_img 內。
```python
# -*- coding: utf-8 -*-
import os
import json
import cv2
import matplotlib.pyplot as plt

for i in os.listdir("json"):
    with open("json/" + i, encoding="utf-8") as f:
          
        data = json.load(f)

        #輸出縮排Json
        jsonData_sort = json.dumps(data, sort_keys = True, indent=4)
        #print(jsonData_sort)
        
        for j in range(0, len(data['shapes'])): 
            if (data['shapes'][j]['group_id'] in [1, 4]): #中文字元，可自己取捨4

                a = data['shapes'][j]['label']
                a = str(a)
                
                x_list, y_list = [], []
                for points in range(0, 4):
                    x_list.append(data['shapes'][j]['points'][points][0])
                    y_list.append(data['shapes'][j]['points'][points][1])         
                xmax, xmin = max(x_list), min(x_list)
                ymax, ymin = max(y_list), min(y_list)
                
                if (ymin < 0):
                    ymin = 0
                if (xmin < 0):
                    xmin = 0
          
                img_ori = cv2.imread("img/" + data['imagePath'])   

                img_after = img_ori[ymin:ymax, xmin:xmax]

                cv2.imencode('.jpg', img_after)[1].tofile("part_img/" + a + "_" + data['imagePath'][:-4] + "_" + str(j) + ".jpg")

               
                
                del x_list, y_list
        
```

<div id="find_numletter_and_writexml"></div>

### **前處理-手動標記英文數字的public data**
### find_numletter_and_writexml.py
&ensp;&ensp;&ensp;&ensp;這部分我們會先挑選出照片中有英文數字的圖片 (groupid = 2, 3)，然後對照其官方給的座標位置，使用 labelimg 進行人工標記，大約取九百張才補齊各英文數字大小寫都超過 20 個，而這部分的資料也同時給英文數字字串及混和字串偵測訓練模型使用。  
&ensp;&ensp;&ensp;&ensp;首先創建一個新資料夾 xml_all，該程式會將原本官方的 json 檔案轉成 xml，並只轉換 groupid = 2, 3 的圖片，且裡面會保留中文字元的部分，因此只要將數字英文標記上去，就可以得到只有中英文及數字的標記檔案，做到能夠讓偵測跟分類都能用到的 Xml 標記檔 (因此前面偵測才會要有一個voc轉yolo的步驟)，並存入剛新增的資料夾。
```python
# -*- coding: utf-8 -*-

import os
import shutil
import json
import cv2

json_data = os.listdir('json/')
num = 0
for i in json_data:   
    with open("json/" + i, encoding="utf-8") as f:
        
        data = json.load(f) # 讀取json
        
        target = []
        for j in range(0, len(data['shapes'])): 
            target.append(data['shapes'][j]['group_id'])

        if 2 in target: # 有英文數字字串的圖片
                    
            img = cv2.imread("img/" + data['imagePath'])
            the_width = data['imageWidth']
            the_heigh = data['imageHeight']
            the_depth = img.shape[2]
            
            #XML撰寫準備--------------------------------------
            space = "space.txt"
            foruse = "xml_all/foruse.txt"
            shutil.copy(space, foruse)
            file = open("xml_all/foruse.txt", mode = "w")
            
            write=file.write("<annotation>\n<filename>" + data['imagePath'] + "</filename>\n<size>\n")
            write=file.write("<width>" + str(the_width) + "</width>\n<height>" + str(the_heigh) + "</height>\n")
            write=file.write("<depth>" + str(the_depth) + "</depth>\n</size>\n")
            
            for j in range(0, len(data['shapes'])): 
                
                x_list, y_list = [], []
                
                if (data['shapes'][j]['group_id'] in [1]): # 取出中文字元
                    
                    for points in range(0, 4):
                        x_list.append(data['shapes'][j]['points'][points][0])
                        y_list.append(data['shapes'][j]['points'][points][1])         
                        xmax, xmin = max(x_list), min(x_list)
                        ymax, ymin = max(y_list), min(y_list)
                        
                    write=file.write("<object>\n<name>" + "zh" + "</name>\n<bndbox>\n")
                    write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                    write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
                
                elif (data['shapes'][j]['group_id'] == 4): # 取出中文單字
                    
                    for points in range(0, 4):
                        x_list.append(data['shapes'][j]['points'][points][0])
                        y_list.append(data['shapes'][j]['points'][points][1])         
                        xmax, xmin = max(x_list), min(x_list)
                        ymax, ymin = max(y_list), min(y_list)
                    
                    write = file.write("<object>\n<name>" + "word" + "</name>\n<bndbox>\n")
                    write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                    write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
                
                del x_list, y_list
                
            write = file.write("</annotation>")    
            file.close()
            os.rename("xml_all/foruse.txt", "xml_all/" + data['imagePath'][:-4] + ".xml")
        elif 3 in target: # 有中英文數混合字串的圖片
                    
            img = cv2.imread("img/" + data['imagePath'])
            the_width = data['imageWidth']
            the_heigh = data['imageHeight']
            the_depth = img.shape[2]
            
            #XML撰寫準備--------------------------------------
            space = "space.txt"
            foruse = "xml_all/foruse.txt"
            shutil.copy(space, foruse)
            file = open("xml_all/foruse.txt", mode = "w")
            
            write=file.write("<annotation>\n<filename>" + data['imagePath'] + "</filename>\n<size>\n")
            write=file.write("<width>" + str(the_width) + "</width>\n<height>" + str(the_heigh) + "</height>\n")
            write=file.write("<depth>" + str(the_depth) + "</depth>\n</size>\n")
            
            for j in range(0, len(data['shapes'])): 
                
                x_list, y_list = [], []
                
                if (data['shapes'][j]['group_id'] in [1]): # 取出中文字元
                    
                    for points in range(0, 4):
                        x_list.append(data['shapes'][j]['points'][points][0])
                        y_list.append(data['shapes'][j]['points'][points][1])         
                        xmax, xmin = max(x_list), min(x_list)
                        ymax, ymin = max(y_list), min(y_list)
                        
                    write=file.write("<object>\n<name>" + "zh" + "</name>\n<bndbox>\n")
                    write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                    write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
                
                elif (data['shapes'][j]['group_id'] == 4): # 取出中文單字
                    
                    for points in range(0, 4):
                        x_list.append(data['shapes'][j]['points'][points][0])
                        y_list.append(data['shapes'][j]['points'][points][1])         
                        xmax, xmin = max(x_list), min(x_list)
                        ymax, ymin = max(y_list), min(y_list)
                    
                    write = file.write("<object>\n<name>" + "word" + "</name>\n<bndbox>\n")
                    write = file.write("<xmin>" + str(xmin) +"</xmin>\n<ymin>" + str(ymin) + "</ymin>\n")        
                    write = file.write("<xmax>" + str(xmax) +"</xmax>\n<ymax>" + str(ymax) + "</ymax>\n</bndbox>\n</object>\n")
                
                del x_list, y_list
                
            write = file.write("</annotation>")    
            file.close()
            os.rename("xml_all/foruse.txt", "xml_all/" + data['imagePath'][:-4] + ".xml")             
```



<div id="Class"></div>

### **前處理-從前步驟做好的資料夾進行類別處理**
### Class.py
&ensp;&ensp;&ensp;&ensp;首先新增資料夾 Class，執行後會將剛剛做好的 part_img 內的所有圖片進行移動，放置到 Class 並按照類別放到各自的類別資料夾 (Class/一、Class/你...)，並移除小於五筆的類別，值得注意的是這裡面並不會有多少英文數字的樣本，需另外製作補充。
```python
# -*- coding: utf-8 -*-
import os 
import shutil

cla = []
for i in os.listdir("part_img"):
    WordClass = i[0]
    if WordClass not in cla:
        cla.append(WordClass)
        os.mkdir("Class/" + WordClass)
        shutil.copy("part_img/" + i, "Class/" + WordClass + "/" + i)
    else:
        shutil.copy("part_img/" + i, "Class/" + WordClass + "/" + i)

#刪調資料小於九筆的類別
for j in os.listdir("Class/"):
    os.rename("CLass/" + j, "CLass/" + str(len(os.listdir("Class/" + j))) + "_" + j)
    if (len(os.listdir("Class/" + j)) < 5):
        shutil.rmtree("Class/" + j)
```


<div id="Rxml"></div>

### **前處理-從前步驟做好的英文數字標記檔提取英文數字樣本**
### Rxml.py
&ensp;&ensp;&ensp;&ensp;這支程式是為了將剛剛 labelimg 做出的資料 (數字英文的補充) 取出，執行完後會得到大小寫以及數字資料夾並且都會統一放進前步驟的 Class 資料夾中，值得注意的是由於 window 資料夾名稱是不分大小寫，因此所有小寫資料夾都會加上一個減號以做區別 (a-)。
```python
# -*- coding: utf-8 -*-
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
    
```



<div id="Select"></div>

### **前處理-從前步驟做好的英文數字標記檔提取英文數字樣本**
### Select.py
&ensp;&ensp;&ensp;&ensp;首先創建一個新資料夾 done，將 Class 內的資料進行切割，最後得到訓練 72%、驗證 18%、 測試 10%。
```python
# -*- coding: utf-8 -*-
import os
import shutil
import random

path = "Class/"
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
            shutil.copy("Class/" + i + "/" + fold[data], "done/train/" + i + "/" + fold[data])
            
        else:  
            shutil.copy("Class/" + i + "/" + fold[data], "done/test/" + i + "/" + fold[data])

path = "done/train/"
for i in os.listdir(path):
    num = len(os.listdir(path + i))
    select_list = random.sample(range(0, num), round(num * 0.8))
    fold = os.listdir(path + "/" + i)
    os.mkdir("done/valid/" + i)
    
    for data in range(0, num):
        if data not in select_list:         
            shutil.move(path + i + "/" + fold[data], "done/valid/" + i + "/" + fold[data])
             
```


<div id="Train"></div>

## 分類模型訓練

<div id="ModelTrain"></div>

### **訓練分類模型InceptionResNetV2**
&ensp;&ensp;&ensp;&ensp;一樣把這支程式跟做好的分類訓練資料放置在同一層，並指定資料夾位置 (done)，裡面會出現三個按照比例分配好的)即可，所有參數都是遵照我們比賽的實際環境，但要注意一點是 NUM_CLASSES 請彈性調整成手邊有的資料類別數量；另外還有要注意的就是儲存權重檔案的路徑，也請記得換成符合自己需要。其餘部分的調整可以參考程式碼內註解，預設會訓練60 Epochs，亦可以自行調整。
```python
# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# 資料路徑
DATASET_PATH  = 'done'

# 影像大小
IMAGE_SIZE = (75, 75)

# 影像類別數
NUM_CLASSES = 2277

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 32

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 1

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-resnet50-e150--ori.h5'

# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2, #水平移動
                                   height_shift_range=0.2, #垂直移動
                                   shear_range=0.3, #x,y一個固定平移
                                   channel_shift_range=10, #變色濾鏡
                                   fill_mode='nearest') #延伸填補)

train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 輸出各類別的索引值
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
    

# 以訓練好的 InceptionResNetV2 為基礎來建立模型，
# 捨棄 InceptionResNetV2 頂層的 fully connected layers
net = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
 
# 輸出整個網路結構
#print(net_final.summary())

# load pre-train 預訓練模型也可自行挑整
#net_final.load_weights('D:/場景文字辨識/高階賽/Model/All_model-InceptionResnetV2-e76.h5')

# 訓練模型
for i in range(1,60): #訓練次數
    WEIGHTS_FINAL = "model-InceptionResnetV2-e" + str(i * 2)
    
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
    #                          patience=3, min_lr=0.0000001)
      
    net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)
                        #,callbacks=[reduce_lr])
    if i >= 1:
        # 儲存訓練好的模型 可自行改變路徑
        net_final.save("D:/場景文字辨識/高階賽/Model/All_"+ WEIGHTS_FINAL + ".h5")
```

<div id="Main"></div>

## 主程式

<div id="Classification"></div>

### **分類模型主程式**
### Classification.py
&ensp;&ensp;&ensp;&ensp;執行該程式後，會開始針對
```python
# -*- coding: utf-8 -*-
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB4

import tensorflow as tf
import sys
import numpy as np
import os
import shutil
import random
import time
import csv
import pandas as pd

clss = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z', 'a-', 'b-', 'c-', 'd-', 'e-', 'f-',
        'g-', 'h-', 'i-', 'j-', 'k-', 'l-','m-', 'n-', 'o-', 'p-', 'q-', 'r-', 's-', 't-', 'u-',
         'v-', 'w-', 'x-', 'y-', 'z-',
        '一', '丁', '七', '三', '上', '下', '不', '世', '丘', '丙', '丞', '丟', '並', '中', '丰', '串', '丸', '丹', '主', '丼', '久', '之', '乘', '乙', '九', '也', '乳', '乾', '亀', '了', '事', '二', '互', '五', '井', '亞', '交', '亨', '享', '京', '亭', '亮', '人', '什', '仁', '今', '介', '从', '仔', '仕', '他', '付', '仙', '仟', '代', '令', '以', '仲', '件', '任', '份', '仿', '企', '伊', '伍', '休', '众', '伙', '伯', '估', '伴', '伸', '佈', '位', '低', '住', '佐', '佑', '何', '佛', '作', '你', '佩', '佰', '佳', '使', '來', '例', '供', '依', '便', '係', '促', '俊', '俗', '保', '俠', '信', '修', '俱', '倆', '倉', '個', '倍', '們', '倒', '候', '借', '倪', '倫', '值', '假', '偉', '偕', '做', '停', '健', '側', '傅', '傑', '傘', '備', '傢', '傳', '傷', '像', '僑', '價', '儀', '億', '儒', '優', '儲', '儷', '元', '兄', '充', '兆', '先', '光', '克', '免', '兒', '兔', '入', '內', '全', '兩', '八', '公', '六', '共', '兵', '其', '具', '典', '兼', '冊', '再', '冒', '冠', '冬', '冰', '冷', '凌', '凍', '凝', '凡', '凰', '凱', '出', '刀', '刁', '分', '切', '刈', '刊', '列', '初', '別', '刨', '利', '刮', '到', '制', '刷', '券', '刺', '刻', '剉', '削', '剌', '前', '剪', '割', '創', '劃', '劇', '劉', '劍', '劑', '力', '功', '加', '助', '努', '劵', '勁', '勇', '勒', '動', '務', '勝', '勞', '勢', '勤', '勳', '勵', '勿', '包', '化', '北', '匙', '匠', '匯', '區', '十', '千', '升', '午', '卉', '半', '卍', '卓', '協', '南', '博', '卜', '卡', '卦', '印', '危', '即', '卷', '卸', '厚', '厝', '原', '厲', '去', '叁', '參', '又', '及', '友', '双', '叔', '取', '受', '口', '古', '另', '只', '叫', '召', '叮', '可', '台', '史', '右', '号', '司', '吃', '各', '合', '吉', '吊', '同', '名', '后', '吐', '向', '君', '吧', '含', '吳', '吸', '吾', '呈', '告', '周', '呱', '味', '呷', '呼', '命', '和', '咕', '咖', '咩', '咪', '咳', '品', '哈', '員', '哥', '哩', '唇', '唐', '售', '唯', '唱', '商', '問', '啓', '啟', '啡', '啤', '啦', '善', '喉', '喘', '喜', '喝', '喫', '喬', '單', '喵', '嗎', '嗑', '嗲', '嗽', '嘉', '嘟', '嘴', '噌', '器', '噴', '嚐', '嚕', '嚴', '囉', '囍', '四', '回', '因', '固', '圈', '國', '圍', '園', '圓', '圖', '團', '土', '在', '地', '圾', '址', '均', '坊', '坐', '坑', '坡', '坤', '坦', '坪', '垂', '垃', '型', '垣', '城', '埔', '埕', '域', '培', '基', '堂', '堃', '堅', '堆', '堡', '堤', '報', '場', '塊', '塑', '塔', '塗', '塩', '填', '塵', '境', '墅', '墊', '墘', '增', '墨', '墾', '壁', '壓', '壞', '壢', '士', '売', '壹', '壺', '壽', '夏', '外', '多', '夜', '夠', '夢', '夥', '大', '天', '太', '夫', '央', '夯', '失', '夷', '夾', '奇', '奈', '奕', '套', '奧', '奪', '女', '奶', '好', '如', '妃', '妍', '妙', '妝', '妮', '妳', '妹', '姆', '姊', '始', '姐', '姑', '姓', '委', '姜', '姨', '姿', '威', '娃', '娘', '娛', '娜', '娥', '婆', '婕', '婚', '婦', '婷', '媒', '媚', '媽', '嫁', '嫂', '嫩', '嬌', '嬰', '子', '孔', '孕', '字', '存', '孝', '季', '学', '孩', '孫', '學', '宅', '宇', '守', '安', '宋', '完', '宏', '宗', '官', '宙', '定', '宜', '宝', '客', '宣', '室', '宥', '宮', '宴', '宵', '家', '宸', '容', '宿', '寄', '密', '富', '寒', '寓', '察', '寢', '實', '寧', '寫', '寬', '寮', '寰', '寵', '寶', '寺', '寿', '封', '射', '將', '專', '尊', '尋', '對', '導', '小', '少', '尖', '尚', '就', '尺', '尼', '尾', '尿', '局', '居', '屈', '屋', '屏', '展', '層', '屬', '山', '岐', '岡', '岩', '岳', '岸', '峰', '島', '峽', '崇', '崎', '崗', '崧', '嵐', '嶺', '嶼', '川', '州', '巢', '工', '左', '巧', '巨', '己', '已', '巴', '巷', '巾', '市', '布', '帆', '希', '帖', '帝', '帥', '師', '席', '帳', '帶', '常', '帽', '幕', '幣', '幫', '干', '平', '年', '幸', '幻', '幼', '庄', '床', '序', '底', '店', '庚', '府', '度', '座', '庫', '庭', '康', '廂', '廈', '廉', '廊', '廖', '廚', '廟', '廠', '廢', '廣', '廬', '廳', '延', '廷', '建', '弄', '式', '弓', '引', '弘', '弟', '張', '強', '彈', '彌', '形', '彥', '彩', '彫', '彭', '彰', '影', '彼', '往', '待', '很', '律', '後', '徐', '徒', '得', '從', '御', '復', '循', '微', '徵', '德', '心', '必', '忌', '志', '忙', '忠', '快', '念', '思', '怡', '急', '性', '怪', '恆', '恒', '恩', '恭', '息', '悅', '悠', '悦', '您', '情', '惟', '惠', '惡', '想', '意', '愛', '感', '慈', '態', '慕', '慢', '慧', '慶', '憲', '應', '懶', '懷', '戀', '成', '我', '戒', '或', '戰', '戲', '戴', '戶', '房', '所', '扁', '扇', '手', '才', '打', '托', '扭', '批', '找', '承', '技', '抄', '把', '抓', '投', '抗', '折', '披', '抹', '抽', '担', '拆', '拉', '拋', '拌', '拍', '拓', '拔', '拖', '招', '拜', '拳', '拷', '拼', '拾', '拿', '持', '指', '按', '挑', '振', '挽', '捏', '捐', '捲', '捷', '掃', '授', '掌', '排', '採', '探', '接', '控', '推', '描', '提', '插', '揚', '換', '握', '揭', '援', '損', '搖', '搜', '搬', '搭', '摩', '撈', '撒', '撞', '撥', '播', '擂', '擇', '擊', '操', '擎', '擴', '攜', '攝', '攤', '支', '收', '改', '放', '政', '故', '效', '敏', '救', '教', '散', '敦', '敬', '敲', '整', '數', '文', '斐', '斑', '斗', '料', '斯', '新', '斷', '方', '於', '施', '旁', '旅', '旋', '族', '旗', '日', '旦', '早', '旭', '旺', '昇', '昌', '明', '易', '昔', '昕', '星', '映', '春', '昭', '是', '昱', '時', '晉', '晚', '晟', '晨', '普', '景', '晴', '晶', '智', '暉', '暖', '暘', '暢', '曉', '曜', '曦', '曲', '更', '書', '曼', '曾', '最', '會', '月', '有', '朋', '服', '朗', '望', '朝', '期', '木', '未', '末', '本', '朱', '朵', '杉', '李', '杏', '材', '村', '杜', '杞', '束', '杯', '杰', '東', '松', '板', '析', '枕', '林', '果', '枝', '架', '柏', '柒', '染', '柔', '柚', '查', '柯', '柳', '柴', '栗', '校', '核', '根', '格', '栽', '桂', '桃', '框', '案', '桌', '桐', '桑', '桔', '桶', '梁', '梅', '條', '梨', '梭', '梯', '械', '梵', '棄', '棉', '棋', '棒', '棟', '棠', '棧', '森', '椅', '植', '椒', '椰', '楊', '楓', '業', '極', '楽', '概', '榔', '榕', '榛', '榜', '榨', '榭', '榮', '構', '槍', '樂', '樓', '標', '模', '樣', '樸', '樹', '樺', '樽', '橋', '橘', '橙', '機', '橡', '橫', '檀', '檜', '檢', '檬', '檯', '檳', '檸', '櫃', '櫥', '櫻', '權', '次', '欣', '款', '歆', '歌', '歐', '歡', '止', '正', '此', '步', '武', '歲', '歸', '殊', '段', '殺', '殼', '殿', '毅', '母', '每', '毒', '比', '毛', '毯', '氏', '民', '氣', '氧', '水', '永', '汁', '求', '汎', '汕', '汗', '江', '池', '污', '汪', '汶', '決', '汽', '沃', '沅', '沉', '沏', '沐', '沒', '沖', '沙', '沛', '沫', '河', '油', '治', '泉', '泌', '泓', '法', '泡', '波', '泥', '注', '泰', '泳', '泵', '洋', '洗', '洛', '洞', '津', '洪', '洱', '洲', '活', '洽', '派', '流', '浦', '浩', '浪', '浮', '浴', '海', '消', '涎', '涮', '液', '涵', '涼', '淇', '淋', '淑', '淘', '淡', '淨', '深', '淳', '淺', '添', '清', '減', '渡', '測', '港', '渴', '游', '湖', '湘', '湯', '湾', '源', '準', '溢', '溪', '溫', '滋', '滑', '滴', '滷', '滾', '滿', '漁', '漂', '漆', '漏', '演', '漢', '漫', '漾', '漿', '潔', '潛', '潢', '潤', '潭', '潮', '澄', '澎', '澡', '澤', '澳', '濃', '濕', '濟', '濰', '濱', '瀚', '灘', '灣', '火', '灰', '灶', '灸', '炎', '炒', '炙', '炫', '炭', '炸', '為', '烈', '烏', '烘', '烤', '烹', '焊', '焗', '焙', '無', '焢', '焦', '然', '焿', '煉', '煌', '煎', '煙', '煤', '照', '煮', '煲', '熊', '熟', '熬', '熱', '燈', '燉', '燒', '燕', '燙', '營', '燥', '燦', '燭', '燴', '燻', '爆', '爌', '爐', '爪', '爭', '爵', '父', '爸', '爹', '爺', '爽', '爾', '牆', '片', '版', '牌', '牙', '牛', '牧', '物', '特', '犬', '狀', '狂', '狗', '狠', '猛', '猴', '獅', '獎', '獨', '獲', '獸', '獻', '玄', '率', '玉', '王', '玖', '玩', '玫', '玲', '玻', '珈', '珊', '珍', '珠', '班', '現', '球', '理', '琦', '琪', '琲', '琳', '琴', '瑋', '瑚', '瑜', '瑞', '瑟', '瑤', '瑩', '瑪', '瑰', '瑾', '璃', '璇', '璉', '璜', '璞', '環', '璽', '瓏', '瓜', '瓦', '瓶', '瓷', '甕', '甘', '甜', '生', '產', '用', '田', '由', '甲', '申', '男', '甸', '町', '界', '留', '畢', '番', '畫', '異', '當', '疆', '疙', '疫', '疼', '疾', '病', '症', '痘', '痛', '痠', '痧', '瘋', '瘦', '瘩', '療', '癌', '癒', '癮', '登', '發', '白', '百', '的', '皆', '皇', '皮', '盅', '盆', '盈', '益', '盒', '盛', '盟', '盡', '監', '盤', '盧', '目', '直', '相', '省', '眉', '看', '真', '眠', '眷', '眼', '眾', '睡', '督', '睫', '睿', '瞳', '知', '短', '矯', '石', '砂', '研', '破', '硬', '硯', '碁', '碑', '碗', '碟', '碧', '碩', '碳', '確', '碼', '磁', '磅', '磚', '磨', '礎', '礙', '礦', '示', '社', '祈', '祐', '祖', '祝', '神', '祥', '票', '祺', '祿', '禁', '禎', '福', '禧', '禪', '禮', '禹', '禾', '秀', '私', '秉', '秋', '科', '秒', '秘', '租', '秤', '移', '稅', '程', '種', '稻', '穀', '穌', '積', '穎', '穗', '究', '空', '穿', '窈', '窕', '窗', '窩', '窯', '立', '站', '章', '童', '端', '競', '竹', '笑', '第', '筆', '等', '筋', '筌', '筍', '筒', '策', '筵', '箋', '算', '管', '箱', '節', '範', '篇', '築', '篩', '簡', '簧', '簽', '簾', '籃', '籍', '籠', '米', '籽', '粄', '粉', '粑', '粒', '粗', '粥', '粧', '粵', '粽', '精', '粿', '糕', '糖', '糧', '糬', '糰', '系', '紀', '約', '紅', '紋', '納', '紐', '紓', '純', '紗', '紙', '級', '素', '紡', '索', '紫', '細', '紳', '紹', '終', '組', '結', '絕', '絡', '給', '統', '絲', '經', '綜', '綠', '維', '綱', '網', '綵', '綸', '綺', '綿', '緊', '緑', '線', '緣', '編', '緬', '緯', '練', '緹', '緻', '縣', '縫', '縮', '總', '織', '繕', '繡', '繪', '繳', '繼', '續', '纖', '罐', '罩', '置', '罰', '署', '羅', '羊', '美', '群', '義', '羹', '羽', '羿', '翁', '翅', '翊', '翎', '習', '翔', '翠', '翡', '翰', '翻', '翼', '耀', '老', '考', '者', '而', '耐', '耕', '耳', '耶', '聖', '聚', '聞', '聯', '聰', '聲', '職', '聽', '肉', '肌', '肚', '肝', '股', '肢', '肥', '肩', '肯', '育', '胃', '背', '胎', '胖', '胡', '胤', '胸', '能', '脂', '脆', '脚', '脫', '腊', '腎', '腐', '腔', '腦', '腰', '腳', '腸', '腹', '腿', '膚', '膜', '膝', '膠', '膩', '膳', '膽', '臉', '臘', '臟', '臣', '臥', '臨', '自', '臭', '至', '致', '臺', '臻', '與', '興', '舉', '舊', '舌', '舍', '舒', '舖', '舘', '舞', '航', '般', '舶', '船', '舺', '艇', '艋', '艦', '良', '色', '艾', '芋', '芒', '芙', '芝', '芬', '芭', '芮', '芯', '花', '芳', '芽', '苑', '苓', '苔', '苗', '若', '苦', '英', '茂', '茄', '茅', '茉', '茗', '茱', '茶', '草', '荳', '荷', '莉', '莊', '莎', '莒', '莓', '莫', '菁', '菇', '菈', '菊', '菌', '菓', '菜', '菠', '菩', '華', '菱', '菲', '菸', '萃', '萄', '萊', '萬', '萱', '落', '葉', '著', '葛', '葡', '董', '葱', '葳', '葵', '蒂', '蒔', '蒙', '蒜', '蒞', '蒸', '蓁', '蓄', '蓉', '蓋', '蓬', '蓮', '蔓', '蔔', '蔗', '蔘', '蔡', '蔣', '蔥', '蔬', '蕃', '蕉', '蕭', '蕾', '薄', '薇', '薏', '薑', '薦', '薩', '薪', '薬', '薯', '藍', '藏', '藝', '藤', '藥', '蘆', '蘇', '蘋', '蘭', '蘿', '虎', '處', '號', '虱', '虹', '蚵', '蛋', '蛤', '蜀', '蜂', '蜜', '蝦', '蝶', '融', '螢', '螺', '蟹', '蠔', '蠟', '蠶', '血', '行', '術', '街', '衛', '衡', '衣', '表', '衫', '袋', '被', '裏', '裕', '補', '裝', '裡', '製', '複', '褲', '襪', '襯', '西', '要', '見', '規', '覓', '視', '親', '覺', '覽', '觀', '角', '解', '觸', '言', '訂', '計', '訊', '訓', '託', '記', '訪', '設', '許', '診', '註', '証', '評', '詠', '詢', '試', '詩', '詮', '話', '詹', '誌', '認', '誕', '語', '誠', '說', '誰', '課', '誼', '調', '請', '論', '諦', '諮', '諾', '謙', '講', '謝', '證', '識', '警', '議', '護', '譽', '讀', '變', '讓', '讚', '谷', '豆', '豐', '豚', '象', '豪', '豬', '貓', '貝', '貞', '負', '財', '貢', '貨', '販', '責', '貳', '貴', '買', '貸', '費', '貼', '貿', '賀', '賃', '資', '賓', '賞', '賢', '賣', '質', '賴', '購', '賽', '贈', '贊', '赤', '赫', '走', '起', '超', '越', '趙', '趣', '足', '趴', '跆', '跌', '跑', '距', '跨', '路', '跳', '踏', '蹈', '蹟', '躍', '身', '車', '軋', '軍', '軒', '軟', '載', '輔', '輕', '輛', '輝', '輪', '輸', '轉', '辛', '辣', '辦', '辰', '農', '迎', '近', '迪', '迷', '追', '退', '送', '透', '逗', '這', '通', '速', '造', '逢', '連', '週', '進', '逸', '遇', '遊', '運', '過', '道', '達', '違', '遙', '遠', '遣', '適', '選', '邀', '邁', '還', '邊', '邑', '那', '邦', '邱', '邸', '郎', '部', '郭', '郵', '都', '鄉', '鄧', '鄭', '鄰', '酉', '酌', '配', '酒', '酥', '酪', '酵', '酷', '酸', '醇', '醉', '醋', '醫', '醬', '釀', '采', '釋', '里', '重', '野', '量', '金', '針', '釣', '鈑', '鈴', '鈺', '鉄', '鉅', '鉑', '銀', '銅', '銓', '銘', '銳', '銷', '鋁', '鋒', '鋪', '鋼', '錄', '錢', '錦', '錫', '錶', '鍊', '鍋', '鍍', '鍵', '鍾', '鎖', '鎮', '鎰', '鏈', '鏡', '鐘', '鐵', '鑄', '鑑', '鑫', '鑲', '鑽', '長', '門', '閃', '閉', '開', '閒', '間', '閣', '閱', '闆', '關', '阪', '防', '阿', '陀', '附', '限', '陞', '院', '除', '陪', '陰', '陳', '陵', '陶', '陸', '陽', '隆', '隊', '階', '隔', '際', '障', '隨', '險', '隱', '隻', '雀', '雄', '雅', '集', '雕', '雙', '雜', '雞', '離', '難', '雨', '雪', '雲', '零', '雷', '電', '需', '震', '霖', '霜', '霞', '霧', '露', '霸', '靈', '靑', '青', '靖', '靚', '靜', '非', '靠', '面', '鞋', '韋', '韓', '音', '韻', '響', '頁', '頂', '項', '順', '頌', '預', '頒', '頓', '領', '頤', '頭', '頸', '頻', '顆', '題', '額', '顎', '顏', '願', '類', '顧', '風', '飄', '飛', '食', '飩', '飯', '飲', '飼', '飽', '飾', '餃', '餅', '養', '餐', '餓', '餘', '餛', '餡', '館', '饅', '饋', '饌', '饕', '饗', '首', '香', '馥', '馨', '馬', '駁', '駐', '駕', '駛', '駿', '騎', '騏', '騰', '驗', '驚', '骨', '體', '高', '髮', '鬆', '鬍', '鬚', '鬥', '鬼', '魂', '魏', '魔', '魚', '魠', '魯', '魷', '鮑', '鮮', '鯛', '鯨', '鰻', '鱈', '鱔', '鱻', '鳥', '鳳', '鴛', '鴦', '鴨', '鴻', '鵝', '鵬', '鶴', '鷄', '鹹', '鹽', '鹿', '麒', '麗', '麟', '麥', '麩', '麵', '麻', '黃', '黎', '黑', '黛', '點', '鼎', '鼓', '鼠', '鼻', '齊', '齋', '齒', '齡', '龍', '龐', '龜']

fix = ['a-','b-','c-','d-','e-','f-','g-','h-','i-','j-','k-','l-','m-','n-','o-','p-','q-','r-','s-','t-','u-','v-','w-','x-','y-','z-']

def test(cls_list, net):

    al = 0
    ans = ""
    after = "00001"
    all_ans = []
    for i in os.listdir("outF/"):
        
        al += 1
        
        img = image.load_img( "outF/" + i, target_size=(75, 75))
        
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:3]

        
        if cls_list[top_inds[0]] in fix:    
            cls_list[top_inds[0]] = cls_list[top_inds[0]][:1]

        quiz = i[:5]

        if (quiz == after):
            ans += cls_list[top_inds[0]]
        else:
            
            all_ans.append(ans)
            ans = cls_list[top_inds[0]] 
           
                
        after = quiz
        
    all_ans.append(ans)
    pd.DataFrame.to_csv(pd.DataFrame(all_ans), "outF_IE_46-2.csv", header=False, index=False)


    
# 影像大小
IMAGE_SIZE = (75, 75)

# 影像類別數
NUM_CLASSES = 2277

# 凍結網路層數
FREEZE_LAYERS = 2

start = 2
#類別載入
cls_list = clss
#print(cls_list)
#模組載入

net = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES , activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
# 請自行改變路徑
net_final.load_weights('D:/場景文字辨識/高階賽/Model/model-InceptionResnetV2-e46-2.h5')
test(cls_list, net_final)




```


<div id="afterprocess2"></div>

## 後處理

<div id="add_ans"></div>

### **後處理-將座標 csv 與分類答案 csv 合併**
### add_ans.py
&ensp;&ensp;&ensp;&ensp;要將分類模型輸出的答案 csv 與剛剛前處理好的座標 csv 做合併，變成最後的繳交檔案，詳細可以看程式碼內的註解來改變讀取檔案的方式，最會便會生成可以上傳的資料格式。
```python
# -*- coding: utf-8 -*-
import pandas as pd

#分類答案csv
out_ans = pd.read_csv("out4_IE_46-2.csv", header = None)
#偵測字串座標csv
coor = pd.read_csv("submission_v4.csv", header = None)

print(coor)
out_ans.columns = [9]

#coor = coor.drop(9, axis = 1)

out_ans=pd.concat([coor, out_ans], axis = 1)
out_ans[1] = out_ans[1].astype(str).astype(int)
out_ans[2] = out_ans[2].astype(str).astype(int)
out_ans[3] = out_ans[3].astype(str).astype(int)
out_ans[4] = out_ans[4].astype(str).astype(int)
out_ans[5] = out_ans[5].astype(str).astype(int)
out_ans[6] = out_ans[6].astype(str).astype(int)
out_ans[7] = out_ans[7].astype(str).astype(int)
out_ans[8] = out_ans[8].astype(str).astype(int)
out_ans[9] = out_ans[9].astype(str)

print(out_ans)
print(out_ans.dtypes)
pd.DataFrame.to_csv(out_ans, "submission_out4_IE_46-2.csv", header=False, index=False)
```
