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

### 載入分割圖片的class  用絕對座標切割
from split_image import cut_image
from PIL import Image

#build model
# 載入模型
print('載入預訓練模型中....')
model = load_model('karting_classify.h5')
model.summary()
print('模型載入完成！')

## 載入Label
print('載入預訓練模型中....')
pkl_file = open('labels.pkl', 'rb')
labels = pickle.load(pkl_file)
print('權重載入完成！')
'''
y = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
y = np.array(y)
answer = labels.inverse_transform(y)
print(answer)
'''

## 處理新的資料放進去prediction
data_path = "screenshot\\fullcut215.png"
source = Image.open(data_path)
source = np.array(source)
reg_source = Image.fromarray(source)
reg_source.show()

print('分割圖片中......')
image = cut_image().get8ImageFromSource(source)  ## 224*224*3 each one
print('圖片分割完成！')

element_index = 1
for element in image[:2]:
    data = []
    reg1 = np.array(element)
    data.append(reg1)
    # 圖片正規化
    data = np.array(data, dtype='float') / 255.0

    ans = model.predict(data)
    #print('預測後的資料是: ', ans)

    final_label = np.zeros(19)
    final_label[np.argmax(ans, axis=1)[0]] = 1
    final_label = [final_label]
    final_label = np.array(final_label)
    #print('印出最大的index: ', np.argmax(ans, axis=1))
    #print('data形狀是: ', ans.shape)
    #print(final_label)
    print('第{0}個道具是： '.format(element_index), labels.inverse_transform(final_label))
    element_index += 1