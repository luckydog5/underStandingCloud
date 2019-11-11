from model import unet
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
threshold = 0.9 
"""Predict output shape is (320,480,4). 4 is classes(Fish, Flower, Gravel, Surger)"""


if __name__ == '__main__':
    img_h = 380
    img_w = 240 
    inputs = (380,240,3)
    mm = unet(inputs)

    image_path = ''
    image = load_img(image_path,target_size=(img_h,img_w))
    assert img is None, print('no image')
    x = img_to_array(image)
    x = np.expand_dims(x,axis=0)
    pred = model.Predict(x)[0]
    print(pred.shape)