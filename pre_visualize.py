from model import unet
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import argparse
import cv2 
import colorsys
import random
from keras.preprocessing.image import array_to_img, img_to_array,load_img
from skimage.measure import find_contours
from matplotlib import patches,lines 
from matplotlib.patches import Polygon

"""Predict outoput shape is (320,480,4). 4 is classes(Fish,Flower,Gravel,Surger)"""
threshold = 0.9 
def apply_mask(image,mask,color,alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:,:,c] = np.where(mask==1,image[:,:,c]*(1-alpha)+alpha*color[c]*255,image[:,:,c])
    return image 
def random_colors(N,bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space
    then convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i/N,1,brightness)for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c),hsv))
    random.shuffle(colors)
    return colors
def display_instances(image,boxes,masks,class_ids,class_names,scores=None,title="",figsize=(20,8),ax=None,show_mask=True,show_bbox=True,colors=None,captions=None):
    """
    N: Number of instances
    boxes: [num_instance,(y1,x1,y2,x2,class_id)] in image coordinates.
    masks: [height,width,num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset 
    scores: (optional) confidence scores for each box 
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    auto_show = False
    N = masks.shape[0]
    if not ax:
        _,ax = plt.subplots(1,figsize=figsize)
        auto_show = True 
    # Generate random colors
    colors = colors or random_colors(N)
    # Show area outside image boundaries.
    height,width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10,width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        # Bounding box 
        x1,y1,x2,y2 = boxes[i]
        print("x1,y1,x2,y2 : {}".format(boxes[i]))
        if show_bbox:
            p = patches.Rectangle((x1,y1),x2,y2,linewidth=2,alpha=0.5,linestyle="dashed",edgecolor=color,facecolor='none')
            ax.add_patch(p)
        # Label 
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label,score) if score else label
        else:
            caption = captions[i]
        ax.text(x1,y1+8,caption,color='w',size=11,backgroundcolor="none")

        # Mask 
        #mask = masks[:,:,i]
        mask = masks[i]
        if show_mask:
            masked_image = apply_mask(masked_image,mask,color)
        
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2,mask.shape[1]+2),dtype=np.uint8)
        padded_mask [1:-1,1:-1] = mask 
        contours = find_contours(padded_mask,0.5)
        for verts in contours:
            # Subtract the padding and flip (y,x) to (x,y)
            verts = np.fliplr(verts) -1
            p = Polygon(verts,facecolor="none",edgecolor=color)
            ax.add_patch(p)
    #cc = plt.imshow(image)
    #ax.add_patch(cc)
    masked_image = cv2.addWeighted(image,0.9,masked_image.astype(np.float32),0.1,0)
    ax.imshow(masked_image)
    #ax.imshow(masked_image.astype(np.uint8))
    #ax.imshow(image)
    ############ Try something different...
    """
    _,ay = plt.subplots(1,figsize=figsize)
    ay.set_ylim(height + 10, -10)
    ay.set_xlim(-10,width + 10)
    ay.axis('off')
    ay.set_title(title)
    ay.imshow(image)
    """
    ## For streamlist.. marked plt.imshow()
    #plt.savefig('result.jpg',dpi=200)
    
    #if auto_show:
        #plt.show()
    return masked_image
def parse_arguments():
    parser = argparse.ArgumentParser(description='Some parameters.')
    parser.add_argument(
        "--image_path",
        type=str,
        help="Image path",
        default=""
    ) 
    return parser.parse_args()
def gen_instances(pred):
    """ Generate instances"""
    class_list = ['Fish','Flower','Gravel','Surgar']
    masks = []
    class_names = []
    boxes = []
    for k in range(pred.shape[-1]):
        temp = pred[...,k]
        pred_mask = temp.astype(np.float32)
        pred_mask,num_predict,rects = post_process(pred_mask,0.8,10000)
        if len(rects) > 0:
            for i in range(len(rects)):
                masks.append(pred_mask)
                class_names.append(class_list[k])
                boxes.append(rects[i])
    class_ids = [i for i in range(len(boxes))]
    return masks,class_names,boxes,class_ids 
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
        # cv.__version__ < 4.0
        #_,contours,hierarchy = cv2.findContours(mask_p.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
        pred_mask,num_predict,rects = post_process(pred_mask,0.8,10000)
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
    verbose = False 
    mm = unet(inputs)
    #args = parse_arguments()
    try:
        mm.load_weights("weights-100-0.73.h5")
        print('Load pre_trained weights !!')
    except Exception as e:
        print('Error: {}'.format(e))
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a1b596.jpg"
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a60891.jpg"
    #image_path = "/home/zsh/underStandingCloud/data/train_images/0a20edf.jpg"
    #image_path = args.image_path
    image_path = "testImages/0a8b542.jpg"
    image = load_img(image_path,target_size=(img_h,img_w))
    #assert len(image.shape)==3,print('None image!')

    x = img_to_array(image)
    x /= 255 
    img = x 
    x = np.expand_dims(x,axis=0)
    pred = mm.predict(x)[0]
    print('x shape: {}'.format(x[0].shape))
    print('pred.shape: {}'.format(pred.shape))
    ###############################################
    #predictions,num,rects = post_process(pred,0.8,10000)
    masks,class_names,boxes,class_ids = gen_instances(pred)
    masks = np.array(masks)
    print('masks shape {}'.format(masks.shape))
    if len(class_ids) > 0:
        display_instances(img,boxes,masks,class_ids,class_names)
    #################################################
    
    #img = visualize(img,pred)
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
