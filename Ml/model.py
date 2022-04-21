import cv2
import tensorflow as tf
from PIL.ImageOps import crop
#from keras.metrics import acc
from keras.metrics import acc
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential

import numpy as np
import pandas as pd
import shutil
import time
# import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
# sns.set_style('darkgrid')
from PIL import Image
# from sklearn.metrics import confusion_matrix, classification_report
# from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging



logging.getLogger("tensorflow").setLevel(logging.ERROR)


# function for get confusion matrix and classification report

def print_confumatrix(test_gen, preds, print_code, save_dir, subject):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    filenames = test_gen.filenames
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {}
    error_indices = []
    y_pred = []
    for key, value in class_dict.items():
        new_dict[value] = key

    classes = list(new_dict.values())  # string list of class names
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]  #  integer values
        if pred_index != true_index:  # if a misclassification has occurred
            error_list.append(filenames[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
            msg = '{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('File name', 'Predict Class', 'Actual Class', 'Probability')
            print(msg)
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]
                msg = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i], true_class[i], ' ',
                                                                       prob_list[i])
                print(msg)
                # print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = 'Accuracy is 100 % so no errors to print'
            print(msg)
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)  # list containg how many times a class c had an error
                plot_class.append(value)  # stores the class
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title(' Errors occur by Class on Test ')
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes) <= 30:
        # create a confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        if length < 8:
            fig_width = 8
            fig_height = 8
        else:
            fig_width = int(length * .5)
            fig_height = int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length) + .5, classes, rotation=90)
        plt.yticks(np.arange(length) + .5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clsreport = classification_report(y_true, y_pred, target_names=classes)
    print(" Report:\n----------------------\n", clsreport)







def train_plot(train_data, start_epoch):
    # Plot the training and validation data
    tacc = train_data.history['accuracy']
    tr_dataloss = train_data.history['loss']
    vacc = train_data.history['val_accuracy']
    val_dataloss = train_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(val_dataloss)  # this is the epoch with the lowest validation loss
    val_lowest = val_dataloss[index_loss]
    index_acc = np.argmax(vacc)
    account_high = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tr_dataloss, 'r', label='Training loss')
    axes[0].plot(Epochs, val_dataloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, account_high, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout

    plt.show()













#function to returns a dataframe where the number of samples for any class specified by column is limited to max samples


def trim_img(dataframe, max_size, min_size, column):
    dataframe = dataframe.copy()
    temp_list = []
    groups = dataframe.groupby(column)
    for label in dataframe[column].unique():
        group = groups.get_group(label)
        ex_count = len(group)
        if ex_count > max_size:
            samples = group.sample(max_size, replace=False, weights=None, random_state=123, axis=0).reset_index(
                drop=True)
            temp_list.append(samples)
        elif ex_count >= min_size:
            temp_list.append(group)
    dataframe = pd.concat(temp_list, axis=0).reset_index(drop=True)
    bal = list(dataframe[column].value_counts())
    print(bal)
    return dataframe






# Function for save ml model
def save(save_path, model, modelName, subject, accuracy, img_size, scalar, generator):

    save_id = str(modelName + '-' + subject + '-' + str(acc)[:str(acc).rfind('.') + 3] + '.h5')   #model saving name
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print(msg)
    # now create the class and convert to csv file
    class_dict = generator.class_indices
    height = []
    width = []
    scale = []
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Ind_ser = pd.Series(list(class_dict.values()), name='class_index')
    Cls_ser = pd.Series(list(class_dict.keys()), name='class')
    Height_ser = pd.Series(height, name='height')
    Width_ser = pd.Series(width, name='width')
    Sle_ser = pd.Series(scale, name='scale by')
    class_df = pd.concat([Ind_ser, Cls_ser, Height_ser, Width_ser, Sle_ser], axis=1)
    csv = 'class_dict.csv'
    csv_save = os.path.join(save_path, csv)
    class_df.to_csv(csv_save, index=False)
    print('csv file was saved as ' + csv_save)
    return model_save_loc, csv_save





#preprocess the data set

def preprocessdata(sdir, trsplit, vsplit):
    file_paths = []
    labels = []
    classlist = os.listdir(sdir)
    for klass in classlist:
        classpath = os.path.join(sdir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            file_paths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(file_paths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    # split dataframe into train_df and test_df
    dsplit = vsplit / (1 - trsplit)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=trsplit, shuffle=True, random_state=123, stratify=strat)
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=123, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
    print(train_df['labels'].value_counts())
    return train_df, test_df, valid_df


#dataset path input
sdir = r'C:\Users\c_chamodkaldera\Downloads\archive\IMG_CLASSES'
train_df, test_df, valid_df = preprocessdata(sdir, .8, .1)



#balance Data set samples

def balance_data(train_df, max_samples, min_samples, column, working_dir, image_size):
    train_df = train_df.copy()
    train_df = trim_img(train_df, max_samples, min_samples, column)
    # make directories to store augmented images
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    # create and store the augmented images
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)
    groups = train_df.groupby('labels')  # group by class
    for label in train_df['labels'].unique():  # for looping every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count = len(group)  # getting how many samples there are in this class
        if sample_count < max_samples:  # if the class has less than target number of images
            aug_img_count = 0
            delta = max_samples - sample_count  # amount of augmented images to create
            target_dir = os.path.join(aug_dir, label)  # define write folder
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
    # create aug_df and merge with train_df to create composite training set usc
    aug_fpaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for class_ in classlist:
        classpath = os.path.join(aug_dir, class_)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(class_)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    usc = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    print(list(usc['labels'].value_counts()))
    return usc


max_samples = 1006
min_samples = 0
column = 'labels'
working_dir = r'/'
img_size = (300, 300)
usc = balance_data(train_df, max_samples, min_samples, column, working_dir, img_size)

channels = 3
batch_size = 30
img_shape = (img_size[0], img_size[1], channels)
length = len(test_df)
test_batch_size = \
sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
print('test batch size: ', test_batch_size, '  test steps: ', test_steps)


def scal(img):
    return img  #  scaling is not required


trgen = ImageDataGenerator(preprocessing_function=scal, horizontal_flip=True)
tvgen = ImageDataGenerator(preprocessing_function=scal)
train_gen = trgen.flow_from_dataframe(usc, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                     class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)

valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
classes = list(train_gen.class_indices.keys())
class_count = len(classes)
train_steps = int(np.ceil(len(train_gen.labels) / batch_size))

# Base model
modelName = 'EfficientNetB3'
base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=img_shape,
                                                  pooling='max')
x = base_model.output
x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.45, seed=123)(x)
output = Dense(class_count, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])


#train the model

epochs = 40
patience = 1  # number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3  # number of epochs to wait before stopping training if monitored value does not improve
threshold = .9  # if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor = .5  # factor to reduce lr by
dwell = True  # experimental, if True and monitored metric does not improve on current epoch set  modelweights back to weights of previous epoch
freeze = False  # if true free weights of  the base model
ask_epoch = 5  # inisial number of epoches
batches = train_steps
callbacks = [
    BaseCall(model=model, base_model=base_model, patience=patience, stop_patience=stop_patience, threshold=threshold,
             factor=factor, dwell=dwell, batches=batches, initial_epoch=0, epochs=epochs, ask_epoch=ask_epoch)]

history = model.fit(x=train_gen, epochs=epochs, verbose=0, callbacks=callbacks, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)


# evaluate model
# save the model

train_plot(history, 0)
sub = 'skin disease'
acc = model.evaluate(test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1] * 100
msg = f'accuracy on the test set is {acc:5.2f} %'
print(msg)
gen = train_gen
scale = 1
model_save_loc, csv_save_loc = save(working_dir, model, modelName, sub, acc, img_size, scale, gen)


print_code=0
preds=model.predict(test_gen, steps=test_steps, verbose=1)
print_confumatrix( test_gen, preds, print_code, working_dir, subject )




# function for plot the training data


def train_plot(tr_data, start_epoch):
    # Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    # plt.style.use('fivethirtyeight')
    plt.show()





