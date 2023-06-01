import cv2
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np

class cut_image:
    def get8ImageFromSource(self, source):
        image_result = []

        ### 1
        area = (426, 383, 600, 500)
        # Crop image
        cropped_img = source[383:500, 426:600]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 2
        area = (619, 383, 794, 500)
        cropped_img = source[383:500 , 619:794]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 3
        area = (813, 384, 981, 500)
        cropped_img = source[384:500, 813:981]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 4
        area = (1005, 384, 1176, 500)
        cropped_img = source[384:500 , 1005:1176]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 5
        area = (1196, 384, 1369, 500)
        cropped_img = source[384:500, 1196:1369]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 6
        area = (426, 536, 600, 639)
        cropped_img = source[536:639, 426:600]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 7
        area = (621, 536, 791, 639)
        cropped_img = source[536:639, 621:791]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 8
        area = (813, 536, 983, 639)
        cropped_img = source[536:639, 813:983]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        image2 = np.array(image2)
        image_result.append(image2)

        ### 9
        area = (1006, 536, 1175, 639)
        cropped_img = source[536:639, 1006:1175]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        
        image2 = np.array(image2)
        image_result.append(image2)

        ### 10
        area = (1199, 536, 1369, 639)
        cropped_img = source[536:639, 1199:1369]
        image2 = Image.fromarray(cropped_img)
        newsize = (224, 224)
        image2 = image2.resize(newsize)
        #image2.show()
        image2 = np.array(image2)
        image_result.append(image2)

        
        return image_result



## 處理新的資料放進去prediction
# data_path = "screenshot\\fullcut0.png"
# data = []
# source = Image.open(data_path)
# source = np.array(source)

# # Crop image
# image_arr = source[1199:1369, 536:639]

# print(source.shape)
# # Convert array to image
# image = Image.fromarray(image_arr)
  
# # Display image
# image.show()
