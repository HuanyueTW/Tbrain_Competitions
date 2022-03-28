# ycl7199
# convert VOC format labels to yolo format
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
