import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt 
import albumentations as albu 
from sklearn.model_selection import train_test_split
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
    if verbose:
        image = base_path + train_df['ImageId'][0]
        img = cv2.imread(image)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
    return train_df,mask_count_df

def one_hot_encoding(train_df=None):
    train_ohe_df = train_df[~ train_df['EncodedPixels'].isnull()]
    classes = train_ohe_df['Label'].unique()
    train_ohe_df = train_ohe_df.groupby('ImageId')['Label'].agg(set).reset_index()
    for class_name in classes:
        train_ohe_df[class_name] = train_ohe_df['Label'].map(lambda x: 1 if class_name in x else 0)
    return train_ohe_df   

class DataGenerator(keras.utils.Sequence):
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
            img_path = f"{self.base_path}/im_name"
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
                masks = build_masks(rels,input_shape=self.dim,reshape=self.reshape)
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



def gen(csv):
    train_df,mask_count_df = read_data(csv,verbose)
    train_ohe_df = one_hot_encoding(train_df)
    img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageId'], train_ohe_df.iloc[:, 2:].values)}
    train_ohe_df['Label'].map(lambda x: str(sorted(list(x))))
    train_idx,val_idx = train_test_split(mask_count_df.index,random_state=42,stratify=train_ohe_df['Label'].map(lambda x: str(sorted(list(x)))),test_size=0.2)
    return train_idx,mask_count_df,train_df,val_idx
if __name__ == '__main__':
    csv = 'data/train.csv'
    verbose = True
    train_df,mask_count_df = read_data(csv,verbose)
    train_ohe_df = one_hot_encoding(train_df)
    img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageId'], train_ohe_df.iloc[:, 2:].values)}
    train_ohe_df['Label'].map(lambda x: str(sorted(list(x))))
    train_idx,val_idx = train_test_split(mask_count_df.index,random_state=42,stratify=train_ohe_df['Label'].map(lambda x: str(sorted(list(x)))),test_size=0.2)
    if verbose:
        print('train_ohe_df head: {}'.format(train_ohe_df.head()))
        print('train length: {}'.format(len(train_idx)))
        print('val length: {}'.format(len(val_idx)))