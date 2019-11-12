from model import unet
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import argparse
from keras.preprocessing.image import array_to_img, img_to_array,load_img

"""Predict outoput shape is (320,480,4). 4 is classes(Fish,Flower,Gravel,Surger)"""
threshold = 0.9 
def parse_argument():
    pass 

if __name__ == '__main__':

    img_h = 320
    img_w = 480
    inputs = (img_h,img_w,3)
    verbose = True
    mm = unet(inputs)
    try:
        mm.load_weights("weights-100-0.73.h5")
        print('Load pre_trained weights !!')
    except Exception as e:
        print('Error: {}'.format(e))
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a1b596.jpg"
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a60891.jpg"
    image_path = "/home/zsh/underStandingCloud/data/train_images/0a20edf.jpg"
    image = load_img(image_path,target_size=(img_h,img_w))
    #assert len(image.shape)==3,print('None image!')

    x = img_to_array(image)
    x /= 255 
    x = np.expand_dims(x,axis=0)
    pred = mm.predict(x)[0]
    print('x shape: {}'.format(x[0].shape))
    print('pred.shape: {}'.format(pred.shape))
  
    Fish = pred[:,:,0]
    Flower = pred[:,:,1]
    Gravel = pred[:,:,2]
    Sugar = pred[:,:,3]
    if verbose:
        plt.figure(figsize=(20,8),dpi=80)
        plt.subplot(231)
        plt.imshow(x[0])
        plt.subplot(232)
        plt.imshow(Fish)
        plt.subplot(233)
        plt.imshow(Flower)
        plt.subplot(234)
        plt.imshow(Gravel)
        plt.subplot(235)
        plt.imshow(Sugar)
        
        plt.show()
