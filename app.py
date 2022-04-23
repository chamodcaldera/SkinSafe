import base64
import secrets
from datetime import date
from dateutil import parser
import keras
from flask import Flask, request, render_template, flash, redirect, jsonify, url_for
import cv2
import numpy as np
import pandas as pd
from PIL.ImageOps import crop
from matplotlib import pyplot as plt
from pyasn1.compat.octets import null
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import os
import shutil
from PIL import Image
from werkzeug.utils import secure_filename

# Store this code in 'app.py' file
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash
# from flask_mysqldb import MySQL
# import MySQLdb.cursors
# import pymysql
import re
import os


from dbConnection import *
from setUp import connection
from werkzeug.security import generate_password_hash, check_password_hash
from saveImage import insertBLOB

connection()
app = Flask(__name__)
# app.secret_key = 'dd hh'
# secret = secrets.token_urlsafe(32)
#
# app.secret_key = secret
app.secret_key = 'super secret key'
# app.config['SESSION_TYPE'] = 'filesystem'
# sess.init_app(app)
UPLOAD_FOLDER_TEST=r'H:\University of westminister\Level 5\SDGP\flaskProject\TestReports'
UPLOAD_FOLDER_PRESS = r'H:\University of westminister\Level 5\SDGP\flaskProject\presImg'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PRESS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_TEST

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


# app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




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


@app.route('/chanellpage',methods=['Get','POST'])
def chan():
    return render_template("channelling.html")

@app.route('/docDash',methods=['Get','POST'])
def docD():
    return render_template("doctors.html")
    # return redirect(url_for('doctor_search'))


@app.route('/prescriptionPage',methods=['Get','POST'])
def presPg():
    return render_template("prescription.html")

@app.route('/adminDashboard',methods=['Get','POST'])
def adminDashboardRedirect():
    if 'loggedin' in session:

        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM Doctor')
        noDoc=cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM User')
        noPat = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM Channelling WHERE status LIKE "%Pending%"')
        noApp = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM Channelling')
        totApp = cursor.fetchone()[0]

        return render_template('admindashboard.html',noDoc=noDoc,noPat=noPat,noApp=noApp,totApp=totApp)

@app.route('/login', methods=['GET', 'POST'])
def login_new():
    msg = ''
    try:
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            conn = mysqldb.connect()
            cursor = conn.cursor()
            if request.form.get('admin') == 'on':
                cursor.execute('SELECT * FROM Admin WHERE email = %s ', (username))
                account = cursor.fetchone()
                if account:
                    # check = check_password_hash(account[4], password)
                    if account[4] == password:
                        session['loggedin'] = True
                        session['id'] = account[0]
                        session['username'] = account[3]
                        msg = 'Logged in successfully !'
                        return redirect(url_for('adminDashboardRedirect'))
                    else:
                        msg = 'Incorrect username / password !'

                else:
                    msg = 'Account dose not exist From this email address '
            else:
                cursor.execute('SELECT * FROM User WHERE email = %s ', (username))
                account = cursor.fetchone()
                if account:
                    check = check_password_hash(account[5], password)
                    if check:
                        session['loggedin'] = True
                        session['id'] = account[0]
                        session['username'] = account[4]
                        msg = 'Logged in successfully !'
                        return render_template('main.html')
                    else:
                        msg = 'Incorrect username / password !'

                else:
                    msg = 'Account dose not exist From this email address '

        return render_template('login.html', msg=msg)

    except Exception as e:
        print(e)


#user Logout
@app.route('/logout')
def logout():
    try:
        pdfList = os.listdir('TestReports')
        testReportsList = ['./static/TestReports/' + image for image in pdfList if
                           ("userId{0}".format(str(session['id']))) in image]
        for pdf in testReportsList:
            os.remove(pdf)
        imageList = os.listdir('./static/Prescriptions')
        imagelist = ['./static/Prescriptions/' + image for image in imageList if
                     ("userId{0}".format(str(session['id']))) in image]
        for image in imagelist:
            os.remove(image)
        session.pop('loggedin', None)
        session.pop('id', None)
        session.pop('username', None)
        return render_template("home.html")
    except Exception as e:
        print(e)

# @app.route('/loginAdmin', methods=['GET', 'POST'])
# def login_admin():
#     msg = ''
#     if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
#         username = request.form['username']
#         password = request.form['password']
#         conn = mysqldb.connect()
#         cursor = conn.cursor()
#         cursor.execute('SELECT * FROM Admin WHERE email = %s ', (username))
#         account = cursor.fetchone()
#         if account:
#             check = check_password_hash(account[4], password)
#             if check:
#                 session['loggedin'] = True
#                 session['id'] = account[0]
#                 session['username'] = account[3]
#                 msg = 'Logged in successfully !'
#                 return render_template('home.html')
#             else:
#                 msg = 'Incorrect username / password !'
#
#         else:
#             msg = 'Account dose not exist From this email address '
#
#     return render_template('admindashboard.html', msg=msg)

@app.route('/regDoc', methods=['GET', 'POST'])
def regDoc():
    return render_template('doctor-register.html')

@app.route('/pat', methods=['GET', 'POST'])
def pat():
    return render_template('patients.html')


# register user
@app.route('/registerUser', methods=['GET', 'POST'])
def register():
    try:
        msg = ''
        if request.method == 'POST' and 'firstname' in request.form and 'lastname' in request.form and 'age' in request.form and 'email' in request.form and 'password' in request.form and 'gender' in request.form and 'address' in request.form and 'mobNo' in request.form:
            firstname = request.form['firstname']
            lastname = request.form['lastname']
            age = request.form['age']
            email = request.form['email']
            password = request.form['password']
            address = request.form['address']
            gender = request.form['gender']
            mobileNo = request.form['mobNo']

            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM User WHERE email = %s', (email,))
            account = cursor.fetchone()
            if account:
                msg = 'Account already exists !'
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = 'Invalid email address !'
            elif containsNumber(firstname) or containsNumber(lastname):
                msg = 'First name and Last name must contain only characters!'
            # elif gender.lower()!='female' or gender.lower()!='female' :
            #     msg = 'Gender'
            elif validate(mobileNo) or len(mobileNo) > 10:
                # must see the validation here
                msg = 'Phone Number must contain only 10 numbers!'

            else:

                # do not save password as a plain text
                _hashed_password = generate_password_hash(password)
                # save edits
                sql = "INSERT INTO User(firstName,lastName, age, email, password, gender, address, mobileNo) VALUES(%s, %s, %s,%s, %s, %s,%s, %s)"
                data = (firstname, lastname, age, email, _hashed_password, gender, address, mobileNo,)
                # conn = mysql.connect()
                # cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                msg = 'You have successfully registered\n Login again! !'
                # return render_template('index.html.html', msg=msg)


        elif request.method == 'POST':

            msg = 'Please fill out the form !'
        return render_template('register.html', msg=msg)

    except Exception as e:
        print(e)


@app.route('/registerUserByAd', methods=['GET', 'POST'])
def register_byAd():
    msg = ''
    if request.method == 'POST' and 'firstname' in request.form and 'lastname' in request.form and 'age' in request.form and 'email' in request.form and 'password' in request.form and 'gender' in request.form and 'address' in request.form and 'mobNo' in request.form:
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        age = request.form['age']
        email = request.form['email']
        password = request.form['password']
        address = request.form['address']
        gender = request.form['gender']
        mobileNo = request.form['mobNo']


        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM User WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif containsNumber(firstname) or containsNumber(lastname):
            msg = 'First name and Last name must contain only characters!'
        # elif gender.lower()!='female' or gender.lower()!='female' :
        #     msg = 'Gender'
        elif validate(mobileNo) or len(mobileNo)>10:
            # must see the validation here
            msg = 'Phone Number must contain only 10 numbers!'

        else:

            # do not save password as a plain text
            _hashed_password = generate_password_hash(password)
            # save edits
            sql = "INSERT INTO User(firstName,lastName, age, email, password, gender, address, mobileNo) VALUES(%s, %s, %s,%s, %s, %s,%s, %s)"
            data = (firstname, lastname, age,email,_hashed_password, gender, address, mobileNo,)
            # conn = mysql.connect()
            # cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            msg = 'You have successfully registered\n Login again! !'
            # return render_template('index.html.html', msg=msg)


    elif request.method == 'POST':

        msg = 'Please fill out the form !'
    return render_template('Patient-register.html',msg=msg)

# doctor register
@app.route('/registerDoctor', methods=['GET', 'POST'])
def registerDoctor():
    try:
        msg = ''
        if request.method == 'POST' and 'firstname' in request.form and 'lastname' in request.form and 'age' in request.form and 'email' in request.form and 'password' in request.form and 'gender' in request.form and 'address' in request.form and 'mobNo' in request.form:
            firstname = request.form['firstname']
            lastname = request.form['lastname']
            age = request.form['age']
            email = request.form['email']
            password = request.form['password']
            address = request.form['address']
            gender = request.form['gender']
            mobileNo = request.form['mobNo']

            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Doctor WHERE docEmail = %s', (email,))
            account = cursor.fetchone()
            if account:
                msg = 'Account already exists !'
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = 'Invalid email address !'
            elif containsNumber(firstname) or containsNumber(lastname):
                msg = 'First name and Last name must contain only characters!'
            elif validate(mobileNo) or len(mobileNo) > 10:
                # must see the validation here
                msg = 'Phone Number must contain only 10 numbers!'

            else:

                # do not save password as a plain text
                _hashed_password = generate_password_hash(password)
                # save edits
                sql = "INSERT INTO Doctor(docFirstName  ,docLastName ,docAge  ,docEmail, docPassword, docGender ,docAddress , docMobileNo ) VALUES(%s, %s, %s,%s, %s, %s,%s, %s)"
                data = (firstname, lastname, age, email, _hashed_password, gender, address, mobileNo,)
                # conn = mysql.connect()
                # cursor = conn.cursor()
                cursor.execute(sql, data)
                conn.commit()
                msg = 'You have successfully registered !'
                # return render_template('index.html.html', msg=msg)


        elif request.method == 'POST':

            msg = 'Please fill out the form !'
        return render_template('doctor-register.html', msg=msg)

    except Exception as e:
        print(e)


# add appointments

# # add appointments
#
# @app.route('/addChannel', methods=['GET','POST'])
# def add_channel():
#
#     if 'loggedin' in session:
#
#         conn = mysqldb.connect()
#         cursor = conn.cursor()
#         if request.method == 'POST' and 'appointment' in request.form:
#             name = request.form['name']
#             if(name==''):
#                 appointment = request.form['appointment']
#                 date,time,meridiem=appointment.split(' ')
#                 hour,min = (int(x) for x in time.split(':'))
#                 month,day, year = (int(x) for x in date.split('/'))
#                 ans = datetime.date(year, month, day)
#                 weekday=ans.strftime("%A")
#                 cursor.execute('SELECT dt.docterId AS docID, dt.day AS day, dt.timeStart AS start, dt.timeEnd AS end, d.docFirstName AS firstName, d.docLastName AS lastName FROM DoctorTimeSlots dt JOIN Doctor d ON dt.docterId = d.docId WHERE day = % s', weekday)
#                 doctorList = cursor.fetchall()
#                 return render_template("channelling.html", doctorList=doctorList)
#
#             elif request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'mobNo' in request.form and 'appointment' in request.form and 'status' in request.form or 'message' in request.form:
#
#                 email = request.form['email']
#                 mobNo = request.form['mobNo']
#                 confirmAppointments = request.form['appointment']
#                 doctorId = request.form['doctorId']
#                 status = 'Pending'
#                 msg = request.form['message']
#                 confirmDate, confirmTime, confirmMeridiem = confirmAppointments.split(' ')
#                 sql = "INSERT INTO Channelling(id,channel_date ,channel_time , docterId , status) VALUES(%s, %s, %s,%s, %s)"
#                 data = ((session['id']), confirmDate, confirmTime, doctorId, status)
#
#                 cursor.execute(sql, data)
#                 conn.commit()
#
#
#         return render_template("channelling.html",)
#     return redirect(url_for('login'))

# dashboard.html load


#display selected doctor

@app.route('/doctorSearch', methods=['GET','POST'])
def doctor_search():

    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            msg=''
            account=[]
            if request.method == 'POST':
                if 'email' in request.form:
                    email = request.form['email']
                    if (request.form.get('button')=='addDoc'):
                        return redirect(url_for('regDoc'))
                    elif (request.form.get('button')=='removeDoc'):
                        return redirect(url_for('doctor_remove',email=email))

                    cursor.execute(
                        'SELECT docId ,docFirstName ,docLastName ,docAge ,docEmail , docGender ,docAddress , docMobileNo  FROM Doctor WHERE docEmail=%s',email)
                    account = cursor.fetchone()
                    if account is None:
                        account = []
                        msg='There is No Doctor Registered From This Email. ('+email+')'

            return render_template("doctors.html", account=account,msg=msg)
        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)

    # display selected doctor

@app.route('/doctorRemove/<email>', methods=['GET', 'POST'])
def doctor_remove(email):

    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            msg = ''
            account = []
            cursor.execute('SELECT * FROM Doctor WHERE docEmail=%s',email)
            account = cursor.fetchone()
            if account is None:
                account = []
                msg = 'There is no doctor registered from this email ' + '(' + email + ')!'
                return render_template("doctors.html", account=account, msg=msg)

            cursor.execute('DELETE FROM Doctor WHERE docEmail=%s',email)
            conn.commit()
            cursor.execute('SELECT docId ,docFirstName ,docLastName ,docAge ,docEmail , docGender ,docAddress , docMobileNo  FROM Doctor WHERE docEmail=%s',email)
            account = cursor.fetchone()
            if account is None:
                account = []
                msg = 'The doctor registered from this email '+'(' + email + ')'+'removed successfully!'
            else:
                msg='Something went wrong while Removing the Doctor Registered from this email ('+email+')'
            return render_template("doctors.html", account=account, msg=msg)
        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)


# patient database management

#display selected doctor

@app.route('/patientSearch', methods=['GET','POST'])
def patient_Search():

    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            msg=''
            account=[]
            if request.method == 'POST':
                if 'email' in request.form:
                    email = request.form['email']
                    if (request.form.get('button')=='addPat'):
                        return redirect(url_for('register_byAd'))
                    elif (request.form.get('button')=='removePat'):
                        return redirect(url_for('patient_remove',email=email))

                    cursor.execute('SELECT id ,firstName  ,lastName  ,age ,email  , gender ,address , mobileNo  FROM User WHERE email=%s',email)
                    account = cursor.fetchone()
                    if account is None:
                        account = []
                        msg='There is No Patient Registered From This Email. ('+email+')'

            return render_template("patients.html", account=account,msg=msg)
        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)

    # display selected doctor

@app.route('/patientRemove/<email>', methods=['GET', 'POST'])
def patient_remove(email):

    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            msg = ''
            account = []
            cursor.execute('SELECT * FROM User WHERE email=%s',email)
            account = cursor.fetchone()
            if account is None:
                account = []
                msg = 'There is no Patient registered from this email ' + '(' + email + ')!'
                return render_template("patient.html", account=account, msg=msg)

            cursor.execute('DELETE FROM User WHERE email=%s',email)
            conn.commit()
            cursor.execute('SELECT * FROM User WHERE email=%s',email)
            account = cursor.fetchone()
            if account is None:
                account = []
                msg = 'The Patient registered from this email '+'(' + email + ')'+'removed successfully!'
            else:
                msg='Something went wrong while Removing the Patient Registered from this email ('+email+')'
            return render_template("patients.html", account=account, msg=msg)
        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)


# prescription display
@app.route("/displayPress")
def displayPress():
    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()

            # cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            # account = cursor.fetchone()
            cursor.execute('SELECT * FROM Prescription WHERE id = % s', (session['id'],))
            record = cursor.fetchone()
            while record is not None:
                storeFilePath = "./static/Prescriptions/userId{0}.img".format(str(session['id'])) + str(
                    record[0]) + ".jpeg"
                print(record)
                base64_img_bytes = record[2]
                # base64_img_bytes = base64_img.encode('utf-8')
                # with open(storeFilePath, "wb") as File:
                #     File.write(record[2])
                #     File.close()
                with open(storeFilePath, 'wb') as file_to_save:
                    decoded_image_data = base64.decodebytes(base64_img_bytes)
                    file_to_save.write(decoded_image_data)
                record = cursor.fetchone()

            # display images

            imageList = os.listdir('./static/Prescriptions')
            imagelist = ['./Prescriptions/' + image for image in imageList if
                         ("userId{0}".format(str(session['id']))) in image]

            return render_template("prescription.html", imagelist=imagelist)

        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)

# display testReport

# prescription display
@app.route("/displayTestReport")
def displayTestReport():
    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()

            # cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            # account = cursor.fetchone()
            cursor.execute('SELECT * FROM TestReports WHERE id = % s', (session['id'],))
            record = cursor.fetchone()
            while record is not None:
                storeFilePath = "./static/TestReports/userId{0}.img".format(str(session['id'])) + str(
                    record[0]) + ".pdf"
                print(record)
                base64_img_bytes = record[2]
                # base64_img_bytes = base64_img.encode('utf-8')
                # with open(storeFilePath, "wb") as File:
                #     File.write(record[2])
                #     File.close()
                with open(storeFilePath, 'wb') as file_to_save:
                    decoded_image_data = base64.decodebytes(base64_img_bytes)
                    file_to_save.write(decoded_image_data)
                record = cursor.fetchone()

            # display images

            pdfList = os.listdir('TestReports')
            testReportsList = ['./TestReports/' + image for image in pdfList if
                               ("userId{0}".format(str(session['id']))) in image]

            return render_template("testReport.html", testReportList=testReportsList)

        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)


#check file extention
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# add prescription

@app.route('/addPress', methods=['GET','POST'])
def addPress():
    try:
        if 'loggedin' in session:
            msg = ''

            if request.method == 'POST':
                if 'image' not in request.files:
                    msg = 'No file input'
                    return render_template("prescription.html", msg=msg)
                file = request.files['image']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    file.filename = "img" + str(session['id']) + ".jpeg"
                    # filename = secure_filename(file.filename)
                    file_location = os.path.join(UPLOAD_FOLDER_PRESS, file.filename)
                    file.save(file_location)

                    # newFile = open(file_location, 'rb').read()
                    # We must encode the file to get base64 string
                    with open(file_location, 'rb') as binary_file:
                        binary_file_data = binary_file.read()
                        base64_encoded_data = base64.b64encode(binary_file_data)
                        uploadFile = base64_encoded_data.decode('utf-8')

                        # uploadFile = base64.b64encode(newFile)
                        # _binaryFile = insertBLOB(file)
                    sql = "INSERT INTO Prescription(id,prescription) VALUES(%s ,%s)"
                    data = ((session['id']), uploadFile,)
                    conn = mysqldb.connect()
                    cursor = conn.cursor()
                    cursor.execute(sql, data)
                    conn.commit()
            # return render_template("prescription.html",msg=msg)
            return redirect(url_for('displayPress'))
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)


#  add test report

# add prescription

@app.route('/addTestReport', methods=['GET','POST'])
def addTestReport():
    try:
        if 'loggedin' in session:
            msg = ''

            if request.method == 'POST':
                if 'image' not in request.files:
                    msg = 'No file input'
                    return render_template("testReport.html", msg=msg)
                file = request.files['testReport']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    file.filename = "TestReport" + str(session['id']) + ".pdf"
                    # filename = secure_filename(file.filename)
                    file_location = os.path.join(UPLOAD_FOLDER_TEST, file.filename)
                    file.save(file_location)

                    # newFile = open(file_location, 'rb').read()
                    # We must encode the file to get base64 string
                    with open(file_location, 'rb') as binary_file:
                        binary_file_data = binary_file.read()
                        base64_encoded_data = base64.b64encode(binary_file_data)
                        uploadFile = base64_encoded_data.decode('utf-8')

                        # uploadFile = base64.b64encode(newFile)
                        # _binaryFile = insertBLOB(file)
                    sql = "INSERT INTO TestReports(id,testReports) VALUES(%s ,%s)"
                    data = ((session['id']), uploadFile,)
                    conn = mysqldb.connect()
                    cursor = conn.cursor()
                    cursor.execute(sql, data)
                    conn.commit()
            # return render_template("prescription.html",msg=msg)
            return redirect(url_for('displayPress'))
        return redirect(url_for('login_new'))

    except Exception as e:

        print(e)


@app.route('/main', methods=['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/adminDashboard', methods=['GET','POST'])
def adminDashboard():
    return render_template('admindashboard.html')



# profile display
@app.route("/display")
def displayProfile():
    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            account = cursor.fetchone()
            return render_template("profile.html", tab=0, account=account)
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)

# update profile
@app.route("/updateUser", methods=['GET', 'POST'])
def updateUser():
    try:
        msg = ''
        if 'loggedin' in session:
            if(request.form.get('button')=='cancel'):
                return redirect('main')
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            account = cursor.fetchone()
            if request.method == 'POST' and 'firstname' in request.form or 'lastname' in request.form or 'age' in request.form or 'address' in request.form or 'mobNo' in request.form:
                firstname = request.form['firstname']
                lastname = request.form['lastname']
                age = request.form['age']
                address = request.form['address']
                mobileNo = request.form['mobNo']
                if containsNumber(firstname) or containsNumber(lastname):
                    msg = 'First name and Last name must contain only characters!'
                elif validate(age) or len(age) > 3:
                    msg = "Inser Age Correctly"
                elif validate(mobileNo) or (len(mobileNo) > 10 or len(mobileNo) < 10):
                    # must see the validation here
                    msg = 'Phone Number must contain only 10 numbers!'
                else:

                    sql = 'UPDATE User SET  firstName =% s, lastName =% s, age =% s, address =% s, mobileNo =% s WHERE id =%s '
                    data = (firstname, lastname, age, address, mobileNo, (session['id'],),)
                    cursor.execute(sql, data)
                    conn.commit()
                    msg = 'You have successfully updated !'
                    cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
                    account = cursor.fetchone()
            elif request.method == 'POST':
                msg = 'Please fill out the form !'
            return render_template('profile.html', tab=1, account=account, msg=msg)
            # return redirect(url_for('display',msg=msg))
        return redirect(url_for('login_new'))
    except Exception as e:
        print(e)

# update password
@app.route("/updatePassword", methods=['GET', 'POST'])
def updatePassword():
    try:
        msg = ''
        if 'loggedin' in session:
            if (request.form.get('button') == 'cancel'):
                return redirect('main')
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            account = cursor.fetchone()
            if request.method == 'POST' and 'email' in request.form and 'oldPass' in request.form and 'newPass' in request.form and 'finalPass' in request.form:
                email = request.form['email']
                oldPass = request.form['oldPass']
                newPass = request.form['newPass']
                finalPass = request.form['finalPass']
                if (session['username'] == email):
                    conn = mysqldb.connect()
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM User WHERE email = %s ', (session['username']))
                    account = cursor.fetchone()
                    if account:
                        check = check_password_hash(account[5], oldPass)
                        if check:
                            if (newPass == finalPass):
                                _hashed_password = generate_password_hash(newPass)
                                sql = 'UPDATE User SET  password =% s WHERE id =%s '
                                data = (_hashed_password, (session['id'],),)
                                cursor.execute(sql, data)
                                conn.commit()
                                msg = 'You have successfully updated Password !'
                            else:
                                msg = 'Re-Enter new Password Correctly'
                        else:
                            msg = 'Incorrect Password'
                else:
                    msg = 'Enter your email correctly'
            elif request.method == 'POST':
                msg = 'Please fill out the form !'
            return render_template('profile.html', tab=2, account=account, msg=msg)
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)


# display channeling and delete channeling
@app.route('/Channelling', methods=['POST','GET'])
def channelling():
    try:
        if 'loggedin' in session:

            msgChan = ''
            conn = mysqldb.connect()
            cursor = conn.cursor()
            if (request.form.get('button') == 'cancel'):
                return redirect('main')
            cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
            account = cursor.fetchone()

            if request.method == 'POST' and 'channelId' in request.form:
                channelId = request.form['channelId']
                conn = mysqldb.connect()
                cursor = conn.cursor()
                cursor.execute('DELETE FROM Channelling WHERE channelId=%s', channelId)
                conn.commit()
                msg = 'Channel number ' + channelId + ' deleted successfully'

            cursor.execute(
                'SELECT c.name as name, c.channelId As id, c.channel_date As date,  c.status as status, d.docFirstName as fname, d.docLastName As last,dt.timeStart As start, c.channel_time As num FROM Channelling c JOIN Doctor d ON c.doctorId=d.docId JOIN DoctorTimeSlots dt ON d. docId=dt.doctorId WHERE id = % s',
                (session['id'],))
            record = cursor.fetchall()
            return render_template("profile.html", tab=3, account=account, record=record, msg=msgChan)
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)

# testChanneling

@app.route('/addChannel', methods=['GET','POST'])
def add_channel():
    try:
        appNumArray = []
        if 'loggedin' in session:
            msgChan = ''
            conn = mysqldb.connect()
            cursor = conn.cursor()
            finalList = []
            if request.method == 'POST' and 'appointment' in request.form:
                appointmentNumber = request.form.get('appNum')
                name = request.form['name']
                email = request.form['email']
                mobNo = request.form['mobNo']

                if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                    msg = 'Invalid email address !'
                elif containsNumber(name):
                    msg = 'Name must contain only characters!'
                elif validate(mobNo) or len(mobNo) > 10:
                    # must see the validation here
                    msg = 'Phone Number must contain only 10 numbers!'

                if (request.form.get('button') == 'checkOption'):
                    appointment = request.form['appointment']
                    CurrentDate = datetime.now().date()
                    ExpectedDate = parser.parse(appointment).date()
                    if CurrentDate > ExpectedDate:
                        msgChan='Invalid Date! Please Select Valid Date'

                    year, month, day = (int(x) for x in appointment.split('-'))
                    ans = date(year, month, day)
                    weekday = ans.strftime("%A")
                    cursor.execute(
                        'SELECT dt.doctorId AS docID, dt.day AS day, dt.timeStart AS start, dt.timeEnd AS end, d.docFirstName AS firstName, d.docLastName AS lastName FROM DoctorTimeSlots dt JOIN Doctor d ON dt.doctorId = d.docId WHERE day = % s',
                        (weekday))
                    doctorList = cursor.fetchall()

                    for doc in doctorList:
                        doctorListAppointment = list(doctorList)
                        cursor.execute(
                            'SELECT channel_time, channel_date doctorId FROM Channelling WHERE channel_date=%s AND doctorId=%s',
                            (appointment, doc[0]))
                        totApp = cursor.fetchall()
                        if (totApp is None):
                            appointmentNumber = 1
                        else:
                            appointmentNumber = len(totApp) + 1
                        docList = list(doc)
                        docList.append(str(appointmentNumber))
                        doc = tuple(docList)

                        finalList.append(doc)

                    finalList = tuple(finalList)

                    return render_template("Channelling.html", doctorList=finalList, appNumArray=appNumArray, name=name,
                                           appointment=appointment, email=email, mobNo=mobNo, msg=msgChan)

                elif request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'appNum' in request.form and 'mobNo' in request.form and 'appointment' in request.form and 'status' in request.form or 'message' in request.form:

                    confirmDate = request.form['appointment']
                    data = request.form['doctorId']
                    doctorId, appointmentNumber = (int(x) for x in data.split(':'))
                    status = 'Pending'
                    msg = request.form['message']

                    sql = "INSERT INTO Channelling(id,name,channel_date ,channel_time , doctorId , status,message) VALUES(%s, %s, %s,%s, %s,%s,%s)"
                    data = ((session['id']), name, confirmDate, appointmentNumber, doctorId, status, msg,)

                    cursor.execute(sql, data)
                    conn.commit()
                    msgChan = "Successfully Added The Appointment!"

            return render_template("Channelling.html", msg=msgChan)
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)

# update password
@app.route("/updateChanStatus", methods=['GET', 'POST'])
def updateChanStatus():
    try:
        msg = ''
        if 'loggedin' in session:

            conn = mysqldb.connect()
            cursor = conn.cursor()
            if request.method == 'POST' and 'email' in request.form and 'status' in request.form and 'chanId' in request.form:
                email = request.form['email']
                status = upper(request.form['status'])
                chanId = request.form['chanId']
                cursor.execute('SELECT id FROM USER WHERE email=%s',(email,))
                account=cursor.fetchone
                if account:
                    cursor.execute('SELECT * FROM Channelling WHERE channelId=%s AND status NOT LIKE %"COMPLETED"%' (chanId))
                    available=cursor.fetchone
                    if available:
                        if status=='COMPLETED'or status=='EXPIRED':
                            sql = 'UPDATE Channelling SET  status =% s WHERE channelId =%s '
                            data = (status, chanId,)
                            cursor.execute(sql, data)
                            conn.commit()
                            msg = 'Successfully updated Status !'
                        else:
                            msg = 'Check the status again'
                    else:
                        msg = 'There is no Ongoing Records from this Channel Id'
                else:
                    msg = 'There is no User registered from this email'
            elif request.method == 'POST':
                msg = 'Please fill out the form !'
            return render_template('statusUpdate.html',msg=msg)
        return redirect(url_for('login_new'))

    except Exception as e:
        print(e)

# add doctor time slots

# register user
@app.route('/docTime', methods=['GET', 'POST'])
def docTime():
    try:
        msg = ''
        if request.method == 'POST' and 'day' in request.form and 'docId' in request.form and 'timeStart' in request.form and 'timeEnd' in request.form :
            day = request.form['day'].title()
            docId = request.form['docId']
            timeStart = request.form['timeStart']
            timeEnd = request.form['timeEnd']

            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Doctor WHERE docId = %s', (docId,))
            account = cursor.fetchone()
            if account:
                # save edits
                sql = "INSERT INTO DoctorTimeSlots(docotrId,day,timeStart,timeEnd) VALUES(%s, %s, %s,%s)"
                data = (docId,day,timeStart,timeEnd)
                cursor.execute(sql, data)
                conn.commit()
                msg = 'Successfully recorded!'
                # return render_template('index.html.html', msg=msg)


        elif request.method == 'POST':

            msg = 'Please fill out the form !'
        return render_template('docTime.html', msg=msg)

    except Exception as e:
        print(e)

def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False
def validate(value):
    for character in value:
        if character.isdigit():
            return False
    return True


if __name__ == '__main__':




    app.run(host="localhost", port=int("5000"))
