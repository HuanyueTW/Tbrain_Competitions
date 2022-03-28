# ycl7199
# transfer json format label to yolo format
import json
import os

# convert coordinates from absolute coordinates to ratio 
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


        all_str = str_ch + str_en + mix # select string types to save in yolo format txt file
        ### save yolo format txt file to output folder
        save_file_name_all = os.path.join("annotations_str", json_filename.replace("json", "txt")) # output folder name  
        print("\n".join(all_str), file= open(save_file_name_all, "w"))


    # print('process done')      
                
json_folder = 'json' # input folder
file_list = os.listdir(json_folder)
print(file_list)
for i, filename in enumerate(file_list):
    json2yolo(filename)
    print('{}/{}'.format(i, len(file_list)))
