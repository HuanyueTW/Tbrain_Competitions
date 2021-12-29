# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:18:05 2021

@author: brian
"""

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
for i in range(1,60):  #訓練次數
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






