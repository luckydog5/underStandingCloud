from model import unet
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import argparse
import cv2 
from keras.preprocessing.image import array_to_img, img_to_array,load_img

"""Predict outoput shape is (320,480,4). 4 is classes(Fish,Flower,Gravel,Surger)"""
threshold = 0.9 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Some parameters.')
    parser.add_argument(
        "--image_path",
        type=str,
        help="Image path",
        default=""
    ) 
    return parser.parse_args()
def post_process(probability,threshold,min_size):
    """
    Post processing of each predicted mask, components with lesser
    number of pixels than 'min_size' are ignored

    """
    rects = []
    mask = cv2.threshold(probability,threshold,1,cv2.THRESH_BINARY)[1]
    num_component,component = cv2.connectedComponents(mask.astype(np.uint8))
    #predictions = np.zeros((350,525),np.float32)
    predictions = np.zeros((320,480),np.float32)
    num = 0
    for c in range(1,num_component):
        p = (component == c)
        #print("p.sum(): {}".format(p.sum()))
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    if num > 0:
        mask_p = predictions.copy()
        contours,hierarchy = cv2.findContours(mask_p.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours,key=cv2.contourArea,reverse=True)[:num]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            rects.append((x,y,w,h))
            print('rect {}'.format((x,y,w,h)))
    return predictions,num,rects
def visualize(img,mask):
    color_list = [(0,0,255),(0,255,0),(255,0,0),(255,100,200)]
    class_list = ['Fish','Flower','Gravel','Surger']
    for k in range(mask.shape[-1]):
        temp = mask[...,k]
        pred_mask = temp.astype(np.float32)
        pred_mask,num_predict,rects = post_process(pred_mask,0.9,10000)
        if len(rects) > 0:
            for rect in rects:
                x,y,w,h = rect 
                cv2.rectangle(img,(x,y),(x+w,y+h),color_list[k],1)
                cv2.putText(img,class_list[k],(x+20,y+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[k], lineType=cv2.LINE_AA)
        else:
            continue
    return img 
if __name__ == '__main__':

    img_h = 320
    img_w = 480
    inputs = (img_h,img_w,3)
    verbose = True
    mm = unet(inputs)
    args = parse_arguments()
    try:
        mm.load_weights("weights-100-0.73.h5")
        print('Load pre_trained weights !!')
    except Exception as e:
        print('Error: {}'.format(e))
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a1b596.jpg"
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a60891.jpg"
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a20edf.jpg"
    image_path = args.image_path
    image = load_img(image_path,target_size=(img_h,img_w))
    #assert len(image.shape)==3,print('None image!')

    x = img_to_array(image)
    x /= 255 
    img = x 
    x = np.expand_dims(x,axis=0)
    pred = mm.predict(x)[0]
    print('x shape: {}'.format(x[0].shape))
    print('pred.shape: {}'.format(pred.shape))
    img = visualize(img,pred)
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
        plt.figure(figsize=(20,8),dpi=80)
        plt.imshow(img)
        plt.show()
