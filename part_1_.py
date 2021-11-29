# Import the necessary packages

import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from google.colab.patches import cv2_imshow
pd.set_option('display.max_columns', 20)

keyfacial_df = pd.read_csv('data.csv')
# print(keyfacial_df)
# print(keyfacial_df.isnull().sum())
# print(keyfacial_df['Image'].shape)
# Then convert this into numpy array using np.fromstring and convert the obtained 1D array into 2D array of shape (96, 96)

keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))
# print(keyfacial_df['Image'])
# print(keyfacial_df['Image'][0].shape)
# print(keyfacial_df.describe())
# print(keyfacial_df['right_eye_center_x'].max())
# print(keyfacial_df['right_eye_center_x'].min())

# keyfacial_df['max'] = keyfacial_df['']
# print(len(keyfacial_df))
i = np.random.randint(1, len(keyfacial_df))
# plt.imshow(keyfacial_df['Image'][i], cmap='gray')
# for j in range(1, 31, 2):
#     plt.plot(keyfacial_df.loc[i][j - 1], keyfacial_df.loc[i][j],'rx')


### Task 3 ##########

fig = plt.figure(figsize=(21, 21))
# for i in range(16):
#     ax = fig.add_subplot(4, 4, i + 1)
#     # i = np.random.randint(1, len(keyfacial_df))
#     image = plt.imshow(keyfacial_df['Image'][i], cmap='gray')
#     for j in range(1, 31, 2):
#         plt.plot(keyfacial_df.loc[i][j - 1], keyfacial_df.loc[i][j], 'rx')



## ogmantation ####

import copy

keyfacial_df_copy = copy.copy(keyfacial_df)
colums = keyfacial_df_copy.columns[:-1]
# print(colums)

keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis=1))
augmmed_df = np.concatenate((keyfacial_df, keyfacial_df_copy))
# print(augmmed_df.shape)
# for i in range(len(colums)):
#     if i%2 == 0 :
#         keyfacial_df_copy[colums[i]]=keyfacial_df_copy[colums[i]].apply(lambda x :96.-float(x))

plt.imshow(keyfacial_df['Image'][i], cmap='gray')
for j in range(1, 31, 2):
    plt.plot(keyfacial_df.loc[i][j - 1], keyfacial_df.loc[i][j],'rx')

# plt.show()
#
# plt.imshow(keyfacial_df_copy['Image'][i], cmap='gray')
# for j in range(1, 31, 2):
#     plt.plot(keyfacial_df_copy.loc[i][j - 1], keyfacial_df_copy.loc[i][j],'rx')



import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))
augmmed_df = np.concatenate((augmmed_df, keyfacial_df_copy))
# print(augmmed_df.shape)

# plt.imshow(keyfacial_df_copy['Image'][i], cmap='gray')
# for j in range(1, 31, 2):
#     plt.plot(keyfacial_df_copy.loc[i][j - 1], keyfacial_df_copy.loc[i][j],'rx')
#
# ###################### quiz ############################################################
# keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda y :np.flip(y,axis=0))
# # augmmed_df  = np.concatenate((keyfacial_df,keyfacial_df_ copy))
# # print(augmmed_df.shape)
# for i in range(len(colums)):
#     if i%2 == 1 :
#         keyfacial_df_copy[colums[i]]=keyfacial_df_copy[colums[i]].apply(lambda y:96.-float(y))
#
# plt.imshow(keyfacial_df_copy['Image'][i], cmap='gray')
# for j in range(1, 31, 2):
#     plt.plot(keyfacial_df_copy.loc[i][j - 1], keyfacial_df_copy.loc[i][j],'rx')

plt.show()

#####################################################################

######## Training Data  ###

img = augmmed_df[:, 30]
img = img / 255.
# print(img[0])
X = np.empty((len(img), 96, 96, 1))

for i in range(len(img)):
    # (96,96,1)
    X[i,] = np.expand_dims(img[i], axis=2)
X = np.asarray(X).astype(np.float32)
# print(X.shape)

y = augmmed_df[:, :30]
y = np.asarray(y).astype(np.float32)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# plt.show()


def res_block(X, filter, stage):
    # Convolutional_block
    X_copy = X

    f1, f2, f3 = filter

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_' + str(stage) + '_conv_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' + str(stage) + '_conv_b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' + str(stage) + '_conv_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_conv_c')(X)

    # Short path
    X_copy = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' + str(stage) + '_conv_copy',
                    kernel_initializer=glorot_uniform(seed=0))(X_copy)
    X_copy = MaxPool2D((2, 2))(X_copy)
    X_copy = BatchNormalization(axis=3, name='bn_' + str(stage) + '_conv_copy')(X_copy)

    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 1
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_' + str(stage) + '_identity_1_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_1_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' + str(stage) + '_identity_1_b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_1_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' + str(stage) + '_identity_1_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_1_c')(X)

    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Identity Block 2
    X_copy = X

    # Main Path
    X = Conv2D(f1, (1, 1), strides=(1, 1), name='res_' + str(stage) + '_identity_2_a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_2_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res_' + str(stage) + '_identity_2_b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_2_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), name='res_' + str(stage) + '_identity_2_c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_' + str(stage) + '_identity_2_c')(X)

    # ADD
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X


input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# 2 - stage
X = res_block(X, filter=[64, 64, 256], stage=2)

# 3 - stage
X = res_block(X, filter=[128, 128, 512], stage=3)
# # 4 - stage
# X = res_block(X, filter= [256,256,1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2, 2), name='Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(4096, activation='relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation='relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation='relu')(X)

model_1_facialKeyPoints = Model(inputs=X_input, outputs=X)
model_1_facialKeyPoints.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='FacialKeyPoints_weights.hdf5', verbose=1, save_best_only=True)

# history = model_1_facialKeyPoints.fit(X_train,y_train,batch_size=33,epochs=15,validation_split=0.05,callbacks=[checkpointer])

model_json = model_1_facialKeyPoints.to_json()
with open('FacialKeyPoints-model.json', 'w') as json_file:
    json_file.write(model_json)

with open('FacialKeyPoints-model.json', 'r') as json_file:
    json_savedmodel = json_file.read()

model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedmodel)
model_1_facialKeyPoints.load_weights('FacialKeyPoints_weights.hdf5')
adam_1 = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(loss='mean_squared_error', optimizer=adam_1, metrics=['accuracy'])

result = model_1_facialKeyPoints.evaluate(X_test, y_test)
print('Accuracy : {} '.format(result[1]))

#
# print(history.history.keys())
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train_loss','val_loss'], loc = 'upper right')
# plt.show()

#### part 2 __ ##########


fasialexperation_df = pd.read_csv('icml_face_data.csv')


# print(fasialexperation_df.head())
# print(fasialexperation_df[' pixels'][0])

def sring2array(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')


def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)


#
fasialexperation_df[' pixels'] = fasialexperation_df[' pixels'].apply(lambda x: sring2array(x))
fasialexperation_df[' pixels'] = fasialexperation_df[' pixels'].apply(lambda x: resize(x))
# print(fasialexperation_df[' pixels'][0])
# print(fasialexperation_df.shape)
# print(fasialexperation_df.isnull().sum())
label_to_text = {0: 'anger', 1: 'disgust', 2: 'sad', 3: 'happiness', 4: 'surprise'}

# fasialexperation_df[' pixels'] = fasialexperation_df[' pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))
# plt.imshow(fasialexperation_df[' pixels'][0], cmap='gray')
# plt.show()


emotions = [0, 1, 2, 3, 4]
# for i in emotions:
#
#     data = fasialexperation_df[fasialexperation_df['emotion']==i][:1]
#     img = data[' pixels'].item()
#     plt.figure()
#     plt.title(label_to_text[i])
#     plt.imshow(img,cmap= 'gray')
#     plt.show()

a = fasialexperation_df.emotion.value_counts().index

b = fasialexperation_df.emotion.value_counts()
# print(a,b)
plt.figure(figsize=(10, 10))
sns.barplot(x=a, y=b)

# plt.show()


from tensorflow.keras.utils import to_categorical

# split the dataframe in to features and labels

X = fasialexperation_df[' pixels']
y = to_categorical(fasialexperation_df['emotion'])
# print(x,y)

X = np.stack(X, axis=0)
X = X.reshape(24568, 96, 96, 1)

# print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

# print(X_val.shape,y_val.shape)
# print(X_test.shape,y_test.shape)
X_train = X_train / 255
X_val = X_val / 255
X_val = X_val / 255

trin_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[1.1, 1.5],
    fill_mode='nearest'
)

input_shape = (96, 96, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# 2 - stage
X = res_block(X, filter=[64, 64, 256], stage=2)

# 3 - stage
X = res_block(X, filter=[128, 128, 512], stage=3)
# # 4 - stage
# X = res_block(X, filter= [256,256,1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2, 2), name='Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(5, activation='softmax', name='Dense_final', kernel_initializer=glorot_uniform(seed=0))(X)

model_2_emotion = Model(inputs=X_input, outputs=X, name='Resnet18')
model_2_emotion.summary()
model_2_emotion.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
checkpointer = ModelCheckpoint(filepath='faicalexprestion_weight.hdf5', verbose=1, save_best_only=True)

# history = model_2_emotion.fit(trin_datagen.flow(X_train, y_train, batch_size=64),
#                               validation_data=(X_val, y_val), steps_per_epoch=len(X_train) // 64,
#                               epochs=15, callbacks=[checkpointer, earlystopping])

model_json1 = model_2_emotion.to_json()
with open('faicalexprestion-model.json', 'w') as json_file:
    json_file.write(model_json1)

with open('faicalexprestion-model.json','r') as json_file:
    json_savedmodel_1 =json_file.read()

model_2_emotion = tf.keras.models.model_from_json(json_savedmodel_1)
model_2_emotion.load_weights('faicalexprestion_weight.hdf5')
model_2_emotion.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])



score = model_2_emotion.evaluate(X_test,y_test)
print('test accuracy : {}'.format(score[1]))

# history.history.keys()
# accuracy= history.history['accuracy']
# val_accuracy =history.history['val_accuracy']
# loss =history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(accuracy))
# plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
#
# plt.plot(epochs, loss, 'ro', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()

prediction_calsses = np.argmax(model_2_emotion.predict(X_test),axis=1)
y_true = np.argmax(y_test,axis=1)

print(y_true.shape)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,prediction_calsses)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,cbar=False)



L = 5
W = 5
#
# fig,axes = plt.subplots(L,W,figsize=(24,24))
# axes = axes.ravel()
#
# for i in np.arange(0,L*W) :
#     axes[i].imshow(X_test[i].reshape(96,96),cmap='gray')
#     axes[i].set_title('prediction = {}\n True = {}'.format(label_to_text[prediction_calsses[i]],label_to_text[y_true[i]]))
#     axes[i].axis('off')
#
# plt.subplots_adjust(wspace= 3 )
from sklearn.metrics import classification_report
# print(classification_report(y_true,prediction_calsses))

# plt.show( )

X_test = X_test/ 250


def prediction (X_test):
    df_pridict = model_1_facialKeyPoints.predict(X_test)
    df_emotion = np.argmax(model_2_emotion.predict(X_test),axis=-1)
    df_emotion = np.expand_dims(df_emotion,axis=1)
    df_pridict = pd.DataFrame(df_pridict,columns=colums)
    df_pridict['emotion'] = df_emotion
    return df_pridict


df_predict = prediction(X_test)
print(df_predict.head())

print(keyfacial_df_copy['right_eye_center_x'].max())

fig, axes = plt.subplots(4, 4, figsize =(24,24))

axes = axes.ravel()


for z in range(16):
    axes[z].imshow(X_test[z].squeeze(), cmap='gray')
    axes[z].set_title('prediction = {}\n True = {}'.format(label_to_text[prediction_calsses[z]],label_to_text[y_true[z]]))
    axes[z].axis('off')
    for j in range(1,31,2):
            axes[z].plot(df_predict.loc[z][j - 1], df_predict.loc[z][j], 'rx')

plt.subplots_adjust(wspace= 2 )

plt.show()


import json
import tensorflow.keras.backend as k

def deploy (directory,model):
    MODEL_DIR = directory
    version = 1
    export_path = os.path.join(MODEL_DIR,str(version))
    print('export path  = {} \n ' .format(export_path))

    if os.path.isdir(export_path):
        print('\nAlready saved a Model , cleaning up \n')

    tf.saved_model.save(model,export_path)
    os.environ['MODEL_DIR'] = MODEL_DIR







