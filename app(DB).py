# Store this code in 'app.py' file
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pymysql
import re
import os


from dbConnection import *
from setUp import connection
from werkzeug.security import generate_password_hash, check_password_hash
from saveImage import insertBLOB

connection()

app = Flask(__name__)

app.secret_key = 'your secret key'
# mysql = MySQL()
# MySQL configurations
# app.config['MYSQL_DATABASE_USER'] = 'root'
# app.config['MYSQL_DATABASE_PASSWORD'] = ''
# app.config['MYSQL_DATABASE_DB'] = 'SkinSafe'
# app.config['MYSQL_DATABASE_HOST'] = 'localhost'
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'SkinSafe'
#
# mysql = MySQL(app)
# mysql.init_app(app)


# doctor register
@app.route('/registerDoctor', methods=['GET', 'POST'])
def registerDoctor():
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
        cursor.execute('SELECT * FROM Doctor WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif containsNumber(firstname) or containsNumber(lastname):
            msg = 'First name and Last name must contain only characters!'
        elif validate(mobileNo) or len(mobileNo)>10:
            # must see the validation here
            msg = 'Phone Number must contain only 10 numbers!'

        else:

            # do not save password as a plain text
            _hashed_password = generate_password_hash(password)
            # save edits
            sql = "INSERT INTO Doctor(docFirstName  ,docLastName ,docAge  ,docEmail, docPassword, docGender ,docAddress , docMobileNo ) VALUES(%s, %s, %s,%s, %s, %s,%s, %s)"
            data = (firstname, lastname, age,email,_hashed_password, gender, address, mobileNo,)
            # conn = mysql.connect()
            # cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            msg = 'You have successfully registered !'
            # return render_template('index.html.html', msg=msg)


    elif request.method == 'POST':

        msg = 'Please fill out the form !'
    return render_template('register.html', msg=msg)

# add appointments

@app.route('/addChannel', methods=['POST'])
def add_channel():

    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()

        date = request.form['date']
        time = request.form['time']
        doctorId = request.form['doctorId']
        status = request.form['status']

        sql = "INSERT INTO Chanelling(id,channel_date ,channel_time , docterId , receipt ) VALUES(%s, %s, %s,%s, %s)"
        data = ((session['id']), date, time, doctorId, status,)
        cursor.execute(sql, data)
        conn.commit()
        return render_template("channelingReceipt.html")
    return redirect(url_for('login'))

# add prescription

@app.route('/addPress', methods=['POST'])
def addPress():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'POST' and 'id' in request.form and 'file' in request.form:
            id = request.form['id']
            file = request.form['image']
            _binaryFile = insertBLOB(file)
            sql = "INSERT INTO Prescription(id,prescription) VALUES(%s,%s)"
            data = (id, _binaryFile,)
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
        return render_template("addPresscription.html",msg=msg)
    return redirect(url_for('login'))

# add test reports

@app.route('/addReports', methods=['POST'])
def addReports():
    if 'loggedin' in session:

        msg = ''
        if request.method == 'POST' and 'id' in request.form and 'file' in request.form:
            id = request.form['id']
            file = request.form['file']
            _binaryFile = insertBLOB(file)
            sql = "INSERT INTO TestReports(id,testReports) VALUES(%s,%s)"
            data = (id, _binaryFile,)
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
        return render_template("addReport.html",msg=msg)

    return redirect(url_for('login'))

#  display All users to Admin
@app.route("/displayAllUsers")
def displayAllUsers():
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT firstName , lastName, emai , age , gender, address , mobileNo FROM User')
        account = cursor.fetchone()
        return render_template("displayAllUsers.html", account=account)
    return redirect(url_for('login'))

# display all doctors to admin

@app.route("/displayAllDoctors")
def displayAllDoctors():
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT docFirstName  ,docLastName ,docAge  ,docEmail, docGender ,docAddress , docMobileNo  FROM Doctor')
        account = cursor.fetchone()
        return render_template("displayAllDoctors.html", account=account)
    return redirect(url_for('login'))

# display all appointments to admin

@app.route("/displayAllAppointments")
def displayAppointments():
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM User Channelling')
        account = cursor.fetchone()
        return render_template("allAppointment.html", account=account)
    return redirect(url_for('login'))

# update single appointments

@app.route("/updateStatus", methods=['GET', 'POST'])
def updateStatus():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'POST' and 'channelId' in request.form:
            channelId = request.form['channelId']
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM Channeling WHERE channelId=%s',channelId)
            account = cursor.fetchone()
        return render_template("displayAllUsers.html", account=account)
    return redirect(url_for('login'))

# like main page
@app.route("/index")
def index():
    if 'loggedin' in session:
        return render_template("index.html")
    return redirect(url_for('login'))

# profile display
@app.route("/display")
def displayProfile():
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
        account = cursor.fetchone()
        return render_template("profile.html", tab=0, account=account)
    return redirect(url_for('login'))

# prescription display
@app.route("/displayPress")
def displayPress():

    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Prescription WHERE id = % s', (session['id'],))
        record = cursor.fetchone()
        cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
        account = cursor.fetchone()

        while record is not None:
            num+1
            storeFilePath = "static/imageOutput/userId{0}.img".format(str(session['id'])) + str(num) + ".jpg"

            with open(storeFilePath, "wb") as File:
                File.write(record[2])
                File.close()
            record = cursor.fetchone()
        #display images

        imageList = os.listdir('static/imageOutput')
        imagelist = ['imageOutput/' + image for image in imageList if ("userId{0}".format(str(session['id']))) in image]
        return render_template("profile.html", tab=3, account=account,imagelist=imagelist)
    return redirect(url_for('login'))

# update profile
@app.route("/updateUser", methods=['GET', 'POST'])
def updateUser():
    msg = ''
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
        account = cursor.fetchone()
        if request.method == 'POST' and 'firstname' in request.form or 'lastname' in request.form or 'age' in request.form or 'gender' in request.form or 'address' in request.form or 'mobNo' in request.form:
            firstname = request.form['firstname']
            lastname = request.form['lastname']
            age = request.form['age']
            address = request.form['address']
            gender = request.form['gender']
            mobileNo = request.form['mobNo']
            if containsNumber(firstname) or containsNumber(lastname):
                msg = 'First name and Last name must contain only characters!'
            elif validate(age) or len(age)>3:
                msg="Inser Age Correctly"
            elif validate(mobileNo) or (len(mobileNo) > 10 or len(mobileNo) < 10):
                # must see the validation here
                msg = 'Phone Number must contain only 10 numbers!'
            else:

                sql='UPDATE User SET  firstName =% s, lastName =% s, age =% s, gender =% s, address =% s, mobileNo =% s WHERE id =%s '
                data=(firstname,lastname, age, gender, address, mobileNo,(session['id'],),)
                cursor.execute(sql, data)
                conn.commit()
                msg = 'You have successfully updated !'
                cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
                account = cursor.fetchone()
        elif request.method == 'POST':
            msg = 'Please fill out the form !'
        return render_template('profile.html', tab=1 ,account=account,msg=msg)
        # return redirect(url_for('display',msg=msg))
    return redirect(url_for('login'))

# change password

# update profile
@app.route("/updatePassword", methods=['GET', 'POST'])
def updatePassword():
    msg = ''
    if 'loggedin' in session:
        conn = mysqldb.connect()
        cursor = conn.cursor()
        # cursor.execute('SELECT * FROM User WHERE id = % s', (session['id'],))
        # account = cursor.fetchone()
        if request.method == 'POST' and 'email' in request.form and 'oldPass' in request.form and 'newPass' in request.form and 'finalPass' in request.form:
            email = request.form['email']
            oldPass = request.form['oldPass']
            newPass = request.form['newPass']
            finalPass = request.form['finalPass']
            if(session['username']==email):
                conn = mysqldb.connect()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM User WHERE email = %s ', (session['username']))
                account = cursor.fetchone()
                if account:
                    check = check_password_hash(account[5], oldPass)
                    if check:
                        if(newPass==finalPass):
                            _hashed_password = generate_password_hash(newPass)
                            sql='UPDATE User SET  password =% s WHERE id =%s '
                            data=(_hashed_password,(session['id'],),)
                            cursor.execute(sql, data)
                            conn.commit()
                            msg = 'You have successfully updated Password !'
                        else:
                            msg='Re-Enter new Password Correctly'
                    else:
                        msg='Incorrect Password'
            else:
                msg='Enter your email correctly'
        elif request.method == 'POST':
            msg = 'Please fill out the form !'
        return render_template('profile.html',tab=2,account=account,msg=msg)
    return redirect(url_for('login'))


@app.route('/deleteChannel', methods=['DELETE'])
def delete_channel():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'DELETE' and 'channelId' in request.form:
            channelId = request.form['channelId']
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM Chanelling WHERE channelId=%s',channelId)
            msg='channel deleted successfully'
        return render_template("myChannel.html", msg=msg)
    return redirect(url_for('login'))

# delete user acc
@app.route('/deleteAcc', methods=['DELETE'])
def delete_acc():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'DELETE':
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM User WHERE id=%s',session['id'])
            msg='User deleted successfully'
        return render_template("display.html", msg=msg)
    return redirect(url_for('login'))

# delete doc acc
@app.route('/deleteAcc', methods=['DELETE'])
def delete_doc():
    if 'loggedin' in session:
        msg = ''
        if request.method == 'DELETE':
            id=request.form['docId']
            conn = mysqldb.connect()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM Doctor WHERE id=%s',id)
            msg='Doctor deleted successfully'
        return render_template("doctorRemove.html", msg=msg)
    return redirect(url_for('login'))

# checking



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

if __name__ == "__main__":
    app.run(host="localhost", port=int("5000"))