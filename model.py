from keras import backend as K 
from keras.layers import Input
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.models import Model 
from keras.optimizers import Adam 
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from utils import gen,DataGenerator
import numpy as np 
import json 

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



def dice_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.* intersection + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true,y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection)+smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
    return 1. - score

def bce_dice_loss(y_true,y_pred):
    return binary_crossentropy(y_true,y_pred) + dice_loss(y_true,y_pred)


def unet(input_shape):
    inputs = Input(shape=input_shape)
    'elu---> Exponential linear unit'
    c1 = Conv2D(8,(3,3),activation='elu',padding='same')(inputs)
    c1 = Conv2D(8,(3,3),activation='elu',padding='same')(c1)
    p1 = MaxPooling2D((2,2),padding='same')(c1)

    c2 = Conv2D(16,(3,3),activation='elu',padding='same')(p1)
    c2 = Conv2D(16,(3,3),activation='elu',padding='same')(c2)
    p2 = MaxPooling2D((2,2),padding='same')(c2)

    c3 = Conv2D(32,(3,3),activation='elu',padding='same')(p2)
    c3 = Conv2D(32,(3,3),activation='elu',padding='same')(c3)
    p3 = MaxPooling2D((2,2),padding='same')(c3)

    c4 = Conv2D(64,(3,3),activation='elu',padding='same')(p3)
    c4 = Conv2D(64,(3,3),activation='elu',padding='same')(c4)
    p4 = MaxPooling2D((2,2),padding='same')(c4)

    c5 = Conv2D(64,(3,3),activation='elu',padding='same')(p4)
    c5 = Conv2D(64,(3,3),activation='elu',padding='same')(c5)
    p5 = MaxPooling2D((2,2),padding='same')(c5)

    c55 = Conv2D(128,(3,3),activation='elu',padding='same')(p5)
    c55 = Conv2D(128,(3,3),activation='elu',padding='same')(c55)

    u6 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c55)
    u6 = concatenate([u6,c5])
    c6 = Conv2D(64,(3,3),activation='elu',padding='same')(u6)
    c6 = Conv2D(64,(3,3),activation='elu',padding='same')(c6)

    u71 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c6)
    u71 = concatenate([u71,c4])
    c71 = Conv2D(32,(3,3),activation='elu',padding='same')(u71)
    c61 = Conv2D(32,(3,3),activation='elu',padding='same')(c71)

    u7 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c61)
    u7 = concatenate([u7,c3])
    c7 = Conv2D(32,(3,3),activation='elu',padding='same')(u7)
    c7 = Conv2D(32,(3,3),activation='elu',padding='same')(c7)

    u8 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c7)
    u8 = concatenate([u8,c2])
    c8 = Conv2D(16,(3,3),activation='elu',padding='same')(u8)
    c8 = Conv2D(16,(3,3),activation='elu',padding='same')(c8)

    u9 = Conv2DTranspose(8,(2,2),strides=(2,2),padding='same')(c8)
    u9 = concatenate([u9,c1],axis=3)
    c9 = Conv2D(8,(3,3),activation='elu',padding='same')(u9)
    c9 = Conv2D(8,(3,3),activation='elu',padding='same')(c9)

    outputs = Conv2D(4,(1,1),activation='sigmoid')(c9)
    model = Model(inputs=[inputs],outputs=[outputs])
    return model 
def train(inputs,data):
    model = unet(inputs)
    model.summary()
    train_idx,mask_count_df,train_df,val_idx = data 
    config = Config()
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
    train_eval_generator = DataGenerator(train_idx, 
                                df=mask_count_df, 
                                target_df=train_df, 
                                batch_size=config.batch_size,
                                reshape=(config.height,config.width),
                                augment=False,
                                graystyle=False,
                                shuffle = False,
                                n_channels=config.channels,
                                n_classes=config.n_classes)
    val_generator = DataGenerator(val_idx, 
                                df=mask_count_df, 
                                target_df=train_df, 
                                batch_size=config.batch_size,
                                reshape=(config.height,config.width),
                                augment=False,
                                graystyle=False,
                                shuffle = False,
                                n_channels=config.channels,
                                n_classes=config.n_classes)
    earlystopping = EarlyStopping(monitor='loss',patience=config.es_patience)
    reduce_lr = ReduceLROnPlateau(monitor='loss',patience=config.rlrop_patience,factor=config.decay_drop,min_lr=1e-6)
    checkpoint = ModelCheckpoint(filepath='weights-{epoch:03d}-{loss:.2f}.h5',monitor='loss',save_best_only=False,save_weights_only=True)
    metric_list = [dice_coef]
    callback_list = [earlystopping,reduce_lr,checkpoint]
    optimizer = Adam(lr=config.learning_rate)
    model.compile(optimizer=optimizer,loss=bce_dice_loss,metrics=metric_list)
    checkpoint.set_model(model)
    history = model.fit_generator(train_generator,validation_data=val_generator,callbacks=callback_list,epochs=100,initial_epoch=0,verbose=2)
if __name__ == '__main__':
    csv = 'data/train.csv'
    train_idx,mask_count_df,train_df,val_idx = gen(csv)
    data = (train_idx,mask_count_df,train_df,val_idx)
