import pandas as pd 
import numpy as  np 
import streamlit as st
from os import listdir
from os.path import isfile,join
from PIL import Image
import cv2
from model import train,unet
from utils import gen,DataGenerator
from pre_visualize import gen_instances,display_instances
from keras.preprocessing.image import array_to_img, img_to_array,load_img
img_w = 480
img_h = 320
def pre(image):
    inputs = (img_h,img_w,3)
    mm = unet(inputs)
    try:
        mm.load_weights("weights-100-0.73.h5")
        print('Load pre_trained weights !!')
    except Exception as e:
        print('Error: {}'.format(e))
    image = load_img(image,target_size=(img_h,img_w))
    x = img_to_array(image)
    x /= 255
    img = x
    x = np.expand_dims(x,axis=0)
    pred = mm.predict(x)[0]
    return pred,img 

def main():
    csv = 'data/train.csv'
    train_idx,mask_count_df,train_df,val_idx = gen(csv,False)
    data = (train_idx,mask_count_df,train_df,val_idx)
    
    showpred = 0
    st.sidebar.title("Content")
    st.sidebar.info("Cloud image segmentation!")
    st.sidebar.title("Train Neural Network")
    if st.sidebar.button("Train CNN"):
        train((img_h,img_w,3),data)

    st.sidebar.title("Predict New Images")
    onlyfiles = [f for f in listdir('/home/zsh/underStandingCloud/testImages/') if isfile(join('/home/zsh/underStandingCloud/testImages/',f))]
    imageselect = st.sidebar.selectbox("Pick an image.",onlyfiles)
    if st.sidebar.button("Predict "):
        showpred = 1
        prediction,img = pre("/home/zsh/underStandingCloud/testImages/"+imageselect)
    
    st.title("Origin image")
    st.write("Pick an image from the left. You will be able to view the image.")
    st.write("When you're ready, submit a prediction on the left.")
    st.write("")
    
    image = Image.open("/home/zsh/underStandingCloud/testImages/"+imageselect)
    st.image(image,caption="Let's predict the image!", use_column_width=True)
    if showpred == 1:
        st.title("Segment image")
        st.write("Pick an image from the left. You will be able to view the image.")
        st.write("When you're ready, submit a prediction on the left.")
        st.write("")
        temp = img
        masks,class_names,boxes,class_ids = gen_instances(prediction)
        masks = np.array(masks)
        masked_image = display_instances(temp,boxes,masks,class_ids,class_names)
        image = Image.fromarray(cv2.cvtColor(masked_image.astype(np.uint8),cv2.COLOR_BGR2RGB))  
        st.image(image,caption="final result",use_column_width=True)
if __name__ == '__main__':
    main()