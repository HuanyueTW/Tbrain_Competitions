# ycl7199
# split dataset to train, validation, test part
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
