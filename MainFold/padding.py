# ycl7199
# padding string image to normal ratio
import cv2
import numpy as np
import os
import json

def padding(filename, img_size ,bbox_info):
    ### read string image
    img = cv2.imread(filename)
    old_image_height, old_image_width, channels = img.shape

    new_image_width = img_size
    new_image_height = img_size
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    ### compute canvas center to put string image
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2


    ### copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    
    print('x center = {}, y center = {}'.format(x_center,y_center))
    b_name = filename.split('.')[0].split('\\')[1]
    ### record center point for recovery step
    bbox_info[b_name] = [x_center, y_center]
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    print( filename.split('\\')[-1])
    save_name = filename.split('\\')[-1]
    save_path = os.path.join(folder_name +'_border\\',save_name)
    ### save result
    cv2.imwrite(save_path, result)


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




