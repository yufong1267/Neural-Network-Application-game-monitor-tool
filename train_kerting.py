from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam

from find_path import GetFileList
import cv2
import pickle


from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

#讀取label
p = r'E:\ipynb\kartRider\screenshot\classify'
DirnameList , FilenameList = GetFileList().FileList(path = p)

base_path = 'E:\\ipynb\\kartRider\\screenshot\\classify\\'
label = []
data = []

#讀取dataset
for label_dir in DirnameList:
    print(label)
    #label下去搜尋下面的資料夾
    full_path = base_path + label_dir
    reg_null , data_set = GetFileList().FileList(path = full_path)
    
    for element in data_set:
        #讀取下面所有所有的dataset
        image = load_img(full_path + '\\' + str(element), target_size=(224, 224))
        image = np.asarray(image)
        data.append(image)
        
        label.append(label_dir)

# 圖片正規化
data = np.array(data, dtype='float') / 255.0
label = np.array(label)

# 標籤二值化
lb = LabelBinarizer()
label = lb.fit_transform(label)

from sklearn.model_selection import train_test_split
# 將資料分為 80% 訓練, 20% 測試
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, random_state=8787)

# 資料增強
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

#build model
#建立模型
vgg19_model = keras.applications.vgg19.VGG19()
model = Sequential()
for layer in vgg19_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(len(label[0]) , activation='softmax'))
model.summary()

EPOCHS = 100
LR = 1e-4
BS = 8

opt = Adam(lr=LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 開始訓練
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

#顯示評估結果
#print(model.evaluate(testX, testY))

model.save('karting_classify.h5')
# 保存標籤
f = open('labels.pkl', 'wb')
f.write(pickle.dumps(lb))
f.close()

        