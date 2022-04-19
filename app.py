import keras
from flask import Flask, request, render_template, flash, redirect, jsonify, url_for
import cv2
import numpy as np
import pandas as pd
from PIL.ImageOps import crop
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import os
import shutil
from PIL import Image
from werkzeug.utils import secure_filename

import application
from application import login_new


# check if the directory was created and image stored


def predictor(sdir, csv_path, model_path, crop_image=False):
    # read in the csv file
    global isave
    class_df = pd.read_csv(csv_path)
    img_height = int(class_df['height'].iloc[0])
    img_width = int(class_df['width'].iloc[0])
    img_size = (img_width, img_height)
    scale = class_df['scale by'].iloc[0]
    try:
        s = int(scale)
        s2 = 1
        s1 = 0
    except:
        split = scale.split('-')
        s1 = float(split[1])
        s2 = float(split[0].split('*')[1])
        print(s1, s2)
    path_list = []
    paths = os.listdir(sdir)
    for f in paths:
        path_list.append(os.path.join(sdir, f))
    print(' Model is being loaded- this will take about 10 seconds')
    # model = load_model(model_path)
    model = keras.models.load_model(model_path)

    image_count = len(path_list)
    index_list = []
    prob_list = []
    cropped_image_list = []
    good_image_count = 0
    for i in range(image_count):
        img = cv2.imread(path_list[i])
        if crop_image == True:
            status, img = crop(img)
        else:
            status = True
        if status == True:
            good_image_count += 1
            img = cv2.resize(img, img_size)
            cropped_image_list.append(img)
            img = img * s2 - s1
            img = np.expand_dims(img, axis=0)
            p = np.squeeze(model.predict(img))
            index = np.argmax(p)
            prob = p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count == 1:
        class_name = class_df['class'].iloc[index_list[0]]
        probability = prob_list[0]
        img = cropped_image_list[0]
        plt.title(class_name, color='blue', fontsize=16)
        plt.axis('off')
        plt.imshow(img)
        return class_name, probability
    elif good_image_count == 0:
        return None, None
    most = 0
    for i in range(len(index_list) - 1):
        key = index_list[i]
        keycount = 0
        for j in range(i + 1, len(index_list)):
            nkey = index_list[j]
            if nkey == key:
                keycount += 1
        if keycount > most:
            most = keycount
            isave = i
    best_index = index_list[isave]
    psum = 0
    bestsum = 0
    for i in range(len(index_list)):
        psum += prob_list[i]
        if index_list[i] == best_index:
            bestsum += prob_list[i]
    img = cropped_image_list[isave] / 255
    class_name = class_df['class'].iloc[best_index]
    # plt.title(class_name, color='blue', fontsize=16)
    # plt.axis('off')
    # plt.imshow(img)
    return class_name, bestsum / image_count


#
csv_path = r'class_dict.csv'  # path to class_dict.csv
model_path = r'EfficientNetB3-skin disease-85.45.h5'

UPLOAD_FOLDER = r'H:\University of westminister\Level 5\SDGP\flaskProject\images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    msg = ""

    global store_path, p, class_name, pre
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:

            image_file.filename = "img.jpg"  # some custom file name that you want
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)

            image_file.save(image_location)
            working_dir = r'H:\University of westminister\Level 5\SDGP\flaskProject'
            store_path = os.path.join(working_dir, 'storage')
            if os.path.isdir(store_path):
                shutil.rmtree(store_path)
            os.mkdir(store_path)
            # input an image of a melanoma
            img_path = r'H:\University of westminister\Level 5\SDGP\flaskProject\images\img.jpg'
            img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # model was trained on rgb images so convert image to rgb
            file_name = os.path.split(img_path)[1]
            dst_path = os.path.join(store_path, file_name)
            cv2.imwrite(dst_path, img)
            # check if the directory was created and image stored
            # msg = os.listdir(store_path)
            class_name, pre = predictor(store_path, csv_path, model_path, crop_image=False)
            msg = f' image is of class {class_name} with a probability of {pre * 100: 6.2f} %'
    return render_template("scanSkin.html", msg=msg)




@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

#
# @app.route('/chanel', methods=['GET', 'POST'])
# def cha():
#     return render_template("channelling.html")


if __name__ == '__main__':
    app.add_url_rule('/login', view_func=application.login_new)
    app.run(host="localhost", port=int("5000"))
