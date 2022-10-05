# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 15:12:56 2020

@author: carlo
"""

##############################################################################
#this is the code for our final model, the way we obtained it is well described 
#in our pdf relation
##############################################################################
#before running this script, run the split_folder.py script



import os
import tensorflow as tf

#I set the seed
SEED = 1234
tf.random.set_seed(SEED)

n_exp = '01'

curr_dir = os.getcwd()                              #current directory
dataset_dir = os.path.join(curr_dir, 'myDataset')   #dataset directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator



valid_perc = 0.2    #I set the percentage of the train dataset going in the validation set
  

#I create training ImageDataGenerator object without doing data augmentation (for the first 5 epochs) 
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    validation_split=valid_perc)
#batch size
bs = 48

#img shape
img_h = 256
img_w = 256

#number of classes
num_classes = 3

#training generator
training_dir = os.path.join(dataset_dir, 'train')
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='training')              #set as training data

#validation generator
valid_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='validation')            #set as validation set


#I create training, validation and test dataset objects
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
valid_dataset = valid_dataset.repeat()



model = tf.keras.Sequential()
start_f = 32                     #starting number of filters

#01 CNN block
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[img_h, img_w, 3]))
model.add(tf.keras.layers.BatchNormalization())    #I apply batch normalization before the relu activation
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
start_f *= 2


#02 CNN block
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None]))
model.add(tf.keras.layers.BatchNormalization())    #I apply batch normalization before the relu activation
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
start_f *= 2


#03 CNN block
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None]))
model.add(tf.keras.layers.BatchNormalization())    #I apply batch normalization before the relu activation
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
start_f *= 2


#04 CNN block
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None]))
model.add(tf.keras.layers.BatchNormalization())     #I apply batch normalization before the relu activation
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu'))
gamma = 0.001   #gamma parameter of weight decay
model.add(tf.keras.layers.Conv2D(filters=start_f, kernel_size=(3,3), strides=(1,1),
                                 padding='valid', input_shape=[None], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(gamma)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


#FC layers
gamma = 0.001   #gamma parameter of weight decay
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(gamma)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))


#Optimization Parameters
#loss
ls = tf.keras.losses.CategoricalCrossentropy()
#learning rate
lr = 1e-4
#optimizer 
optim = tf.keras.optimizers.Adam(learning_rate=lr)
#validation metric
val_metric = ['accuracy']


#I compile the model
model.compile(optimizer=optim, loss=ls, metrics=val_metric)

#I add the callbacks

#I build the directories
from datetime import datetime

fold_name = 'classification_experiments' + n_exp
exps_dir = os.path.join(curr_dir, fold_name)
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_name = 'CNN'
exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

#I add the model checkpoints
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

#I visualize learning on TensorBoard
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

#I implement early stopping
early_stop = True
pat = 10    #patience of the early stopping
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = pat)
callbacks.append(tb_callback)


print(model.summary())




#I train the model for the first 5 epochs without data augmentation
num_epoch = 5
model.fit(x=train_dataset,
          epochs = num_epoch,
          steps_per_epoch = len(train_gen),
          validation_data = valid_dataset,
          validation_steps = len(valid_gen),
          callbacks = callbacks)


#now that the model is trained for 5 epochs, I train it with data augmentation

bs = 32     #I set batch size

#I create training ImageDataGenerator object doing data augmentation
train_data_gen = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=10,
                                    height_shift_range=10,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode='constant',
                                    cval=0,
                                    rescale=1./255,
                                    validation_split=valid_perc)

#train generator
train_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='training')              #set as training data

#validation generator
valid_gen = train_data_gen.flow_from_directory(training_dir,
                                               batch_size=bs,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=SEED,
                                               subset='validation')            #set as validation set


#I create training, validation dataset objects
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
train_dataset = train_dataset.repeat()

valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
valid_dataset = valid_dataset.repeat()

#
tb_dir = os.path.join(exp_dir, 'tb_logs_02')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks = []
callbacks.append(ckpt_callback)
callbacks.append(tb_callback)



#I train the model with data augmentation
num_epoch = 100
model.fit(x=train_dataset,
          epochs = num_epoch,
          steps_per_epoch = len(train_gen),      
          validation_data = valid_dataset,
          validation_steps = len(valid_gen),
          callbacks = callbacks)