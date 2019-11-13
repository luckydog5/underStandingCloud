import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt 
import albumentations as albu 
import keras
import random 
import colorsys
from matplotlib import patches,lines
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split
from sklearn.measure import find_contours
from keras.models import Model 
from model import unet
class Config(object):
    batch_size = 32
    backbone = 'resnet34'
    encoding_weights = 'imagenet'
    activation = 'sigmoid'
    epochs = 30
    learning_rate = 3e-4
    height = 320
    width = 480
    channels = 3
    es_patience = 5
    rlrop_patience = 3
    decay_drop = 0.5
    n_classes = 4 
def np_resize(img,input_shape,graystyle=False):
    """
    Reshape a numpy array, which is input_shape=(height,width),
    as opposed to input_shape=(width,height) for cv2
    """
    height,width = input_shape
    resized_img = cv2.resize(img,(width,height))

    if graystyle:
        resized_img = resized_img[...,None]
    return resized_img

def mask2rle(img):
    """
    img: a mask image, numpy array, 1-mask, 0-background
    Returns run length as string formated

    img.T.flatten()
    image(width * height * channel),
    from width -> height -> channel flatten a one dimension array
    """
    pixels = img.T.flatten()
    # add 0 to the beginning and end of array
    pixels = np.concatenate([[0],pixels,[0]])
    runs = np.where(pixels[1:]!=pixels[:-1])[0] + 1
    # from index 1 select every 2 elements
    # form index 0 select every 2 elements
    # 从1开始每隔2个的数进行重新赋值
    runs[1::2] -= runs[::2]
    return ''.join(str(x) for x in runs)

def rle2mask(rle,input_shape):
    width,height = input_shape[:2]
    mask = np.zeros(width*height).astype(np.uint8)
    array = np.array([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index,start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
    return mask.reshape(height,width).T 
def build_masks(rles,input_shape,reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape,depth))
    else:
        masks = np.zeros((*reshape,depth))
    for i,rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:,:,i] = rle2mask(rle,input_shape)
            else:
                mask = rle2mask(rle,input_shape)
                reshape_mask = np_resize(mask,reshape)
                masks[:,:,i] = reshape_mask

    return masks
def read_data(csv,verbose=False):
    train_df = pd.read_csv(csv)
    base_path = 'data/train_images/'
    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x : x.split('_')[0])
    train_df['Label'] = train_df['Image_Label'].apply(lambda x : x.split('_')[1])
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
    ##############################################################
    ## mask_count_df.........
    mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
    mask_count_df.sort_values('hasMask',ascending=False,inplace=True)
    if verbose:
        print('mask_count_df.shape: {}'.format(mask_count_df.shape))

    #####
    print(train_df.shape)
    print(train_df.head())
    #####
    """
    if verbose:
        image = base_path + train_df['ImageId'][0]
        img = cv2.imread(image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    """
    return train_df,mask_count_df

def one_hot_encoding(train_df=None):
    train_ohe_df = train_df[~ train_df['EncodedPixels'].isnull()]
    classes = train_ohe_df['Label'].unique()
    train_ohe_df = train_ohe_df.groupby('ImageId')['Label'].agg(set).reset_index()
    for class_name in classes:
        train_ohe_df[class_name] = train_ohe_df['Label'].map(lambda x: 1 if class_name in x else 0)
    return train_ohe_df   

#class DataGenerator(keras.utils.Sequence):
class DataGenerator():
    def __init__(self,list_IDs,df,target_df=None,mode='fit',base_path='data/train_images',batch_size=32,dim=(1400,2100),n_channels=3,reshape=None,augment=False,n_classes=4,random_state=42,shuffle=True,graystyle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df 
        self.mode = mode 
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.augment = augment
        self.shuffle = shuffle
        self.random_state = random_state
        self.graystyle = graystyle
        self.on_epoch_end()
        self.n_classes = n_classes
        np.random.seed(self.random_state)
    def __len__(self):
        'Denotes the number of batched per epoch'
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    def __getitem__(self,index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs, real image id
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X = self.__generate_X(list_IDs_batch)
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            if self.augment:
                X,y = self.__augment_batch(X,y)
            return X,y
        elif self.mode == 'predict':
            return X 
        else:
            raise AttributeError('The mode parameters should be set to "fit" or "predict".')
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    def __generate_X(self,list_IDs_batch):
        'Generate data containing batch_size samples'
        if self.reshape is None:
            X = np.empty((self.batch_size,*self.dim,self.n_channels))
        else:
            X = np.empty((self.batch_size,*self.reshape,self.n_channels))
        # Generate data
        for i,ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            #img_path = f"{self.base_path}/im_name"
            img_path = self.base_path + '/' + im_name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if self.reshape is not None:
                img = np_resize(img,self.reshape)

            X[i,] = img

        return X 
    def __generate_y(self,list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size,*self.dim,self.n_classes),dtype=int)
        else:
            y = np.empty((self.batch_size,*self.reshape,self.n_classes),dtype=int)

        for i,ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId']== im_name]
            rles = image_df['EncodedPixels'].values
            if self.reshape is not None:
                masks = build_masks(rles,input_shape=self.dim,reshape=self.reshape)
            else:
                masks = build_masks(rles,input_shape=self.dim)
            y[i,] = masks

        return y 
    def __load_rgb(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img 
    def __random_transform(self,img,masks):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            #albu.RandomRotate90(p=1),
            #albu.RandomBrightness(),
            #albu.ElasticTransform(p=1,distort_limit=2,sigma=120*0.05,alpha_affine=120120*0.03),
            albu.GridDistortion(p=0.5)])

        composed = composition(image=img,mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        return aug_img,aug_masks
    def __augment_batch(self,img_batch,masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,],masks_batch[i,] =self.__random_transform(img_batch[i,],masks_batch[i,])

        return img_batch,masks_batch
    def get_labels(self):
        if self.shuffle:
            images_current = self.list_IDs[:self.len * self.batch_size]
            labels = [img_to_ohe_vector[img] for img in images_current]

        return np.array(labels)



def gen(csv,verbose=False):
    train_df,mask_count_df = read_data(csv,verbose)
    train_ohe_df = one_hot_encoding(train_df)
    img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageId'], train_ohe_df.iloc[:, 2:].values)}
    train_ohe_df['Label'].map(lambda x: str(sorted(list(x))))
    train_idx,val_idx = train_test_split(mask_count_df.index,random_state=42,stratify=train_ohe_df['Label'].map(lambda x: str(sorted(list(x)))),test_size=0.2)
    return train_idx,mask_count_df,train_df,val_idx
def post_process(probability,threshold,min_size):
    """
    Post processing of each predicted mask, components with lesser
    number of pixels than 'min_size' are ignored

    """
    rects = []
    mask = cv2.threshold(probability,threshold,1,cv2.THRESH_BINARY)[1]
    num_component,component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350,525),np.float32)
    num = 0
    for c in range(1,num_component):
        p = (component == c)
        print("p.sum(): {}".format(p.sum()))
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
def sigmoid(x):
    return 1/(1+np.exp(-x))

def extract_layer_output(model,layer_name,x):
    """
    model: load pretrained weights and specify inputs shape
    layer_name: which layer output will you use
    x: image to extract features from
    return: numpy.array(N,H,W,C)
    """
    intermediate_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_model.predict(x)
    return intermediate_output 
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
    colors = list(map(lambda c: clolorsys.hsv_to_rgb(*c),hsv))
    random.shuffle(colors)
    return colors
def display_instances(N,image,boxes,masks,class_ids,class_names,scores=None,title="",figsize=(20,8),ax=None,show_mask=True,show_bbox=True,colors=None,captions=None):
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
        if show_bbox:
            p = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,alpha=0.7,linestyle="dashed",edgecolor=color,facecolor='none')
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
        mask = masks[:,:,i]
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
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()





if __name__ == '__main__':
    csv = 'data/train.csv'
    config = Config()
    verbose = True
    train_df,mask_count_df = read_data(csv,verbose)
    train_ohe_df = one_hot_encoding(train_df)
    img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageId'], train_ohe_df.iloc[:, 2:].values)}
    train_ohe_df['Label'].map(lambda x: str(sorted(list(x))))
    train_idx,val_idx = train_test_split(mask_count_df.index,random_state=42,stratify=train_ohe_df['Label'].map(lambda x: str(sorted(list(x)))),test_size=0.2)
    train_generator = DataGenerator(train_idx, 
                                df=mask_count_df, 
                                target_df=train_df, 
                                batch_size=config.batch_size,
                                reshape=(config.height,config.width),
                                augment=True,
                                graystyle=False,
                                shuffle = True,
                                n_channels=config.channels,
                                n_classes=config.n_classes)
    
    if verbose:
        print('train_ohe_df head: {}'.format(train_ohe_df.head()))
        print('train length: {}'.format(len(train_idx)))
        print('val length: {}'.format(len(val_idx)))
        print("train_generator lengh is ", len(train_generator))
        x,y = train_generator.__getitem__(0)
        im_x = x[2]
        mask_x = y[2]
        print(x.shape,y.shape)
        print(mask_x[:,:,0].shape)
        rectts = []
        color_list = [(0,0,255),(0,255,0),(255,0,0),(255,100,200)]
        class_list = ['Fish','Flower','Gravel','Surger']
        print("y.shape[-1] : {}".format(y.shape[-1]))
        print('y[0][:,:,1] type is {}, its {}'.format(type(y[0][:,:,1]),y[0][:,:,1]))
        if im_x.shape != (350,525):
            xx = cv2.resize(im_x,dsize=(525,350),interpolation=cv2.INTER_LINEAR)
        for k in range(y.shape[-1]):
            print('--k is : {} --'.format(k))
            print("type y[0] is {}".format(type(mask_x)))
            temp = mask_x[...,k].copy()
            #pred_mask = y[0][:,:,k].astype('float32')
            pred_mask = temp.astype(np.float32)
            if pred_mask.shape != (350,525):
                pred_mask = cv2.resize(pred_mask,dsize=(525,350),interpolation=cv2.INTER_LINEAR)
            print('pred_mask shape {}'.format(pred_mask.shape))
            #predd_mask,num_predict,rects = post_process(sigmoid(pred_mask),0.5,25000)
            predd_mask,num_predict,rects = post_process(pred_mask,0.5,0)
            print('num_predict is {}'.format(num_predict))
            print('rects {}'.format(len(rects)))
            if len(rects) > 0:
                for rect in rects:
                    x1,yy,w,h = rect 
                    print('x,y,w,h {}'.format(rect))
                    print('color_list k is {}'.format(color_list[k]))
                    cv2.rectangle(xx,(x1,yy),(x1+w,yy+h),color_list[k],1)
                    cv2.putText(xx,class_list[k],(x1,yy),cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_list[k], lineType=cv2.LINE_AA)
            else:
                continue
        plt.figure(figsize=(20,8),dpi=80)
        plt.imshow(xx)
        
        
        plt.figure(figsize=(20,8),dpi=80)    
        plt.subplot(231)
        plt.imshow(im_x)
        plt.subplot(232)
        plt.imshow(mask_x[:,:,0])
        plt.subplot(233)
        plt.imshow(mask_x[:,:,1])
        plt.subplot(234)
        plt.imshow(mask_x[:,:,2])
        plt.subplot(235)
        plt.imshow(mask_x[:,:,3])
        plt.subplot(236)
        plt.imshow(mask_x)
        plt.show()
        