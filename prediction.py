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
from keras.models import load_model

#build model
# 載入模型
model = load_model('karting_classify.h5')
model.summary()

## 載入Label
pkl_file = open('labels.pkl', 'rb')
labels = pickle.load(pkl_file)
'''
y = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
y = np.array(y)
answer = labels.inverse_transform(y)
print(answer)
'''

## 處理新的資料放進去prediction
data_path = "screenshot\\classify\\魔音x\\403.png"
data = []
image = load_img(data_path, target_size=(224, 224))
image = np.asarray(image)
data.append(image)
# 圖片正規化
data = np.array(data, dtype='float') / 255.0
ans = model.predict(data)
print('預測後的資料是: ', ans)

final_label = np.zeros(19)
final_label[np.argmax(ans, axis=1)[0]] = 1
final_label = [final_label]
final_label = np.array(final_label)
print('印出最大的index: ', np.argmax(ans, axis=1))
print('data形狀是: ', ans.shape)
print(final_label)
print('Label 最後的結果是: ', labels.inverse_transform(final_label))