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

# app.secret_key = 'your secret key'
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


# User LogIn
# @app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login_new():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        conn = mysqldb.connect()
        cursor = conn.cursor()
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

# admin LogIn
# @app.route('/')
@app.route('/loginAdmin', methods=['GET', 'POST'])
def login_admin():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        conn = mysqldb.connect()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Admin WHERE email = %s ', (username))
        account = cursor.fetchone()
        if account:
            check = check_password_hash(account[4], password)
            if check:
                session['loggedin'] = True
                session['id'] = account[0]
                session['username'] = account[3]
                msg = 'Logged in successfully !'
                return render_template('home.html')
            else:
                msg = 'Incorrect username / password !'

        else:
            msg = 'Account dose not exist From this email address '

    return render_template('admindashboard.html', msg=msg)

# register user
@app.route('/registerUser', methods=['GET', 'POST'])
def register():
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
            msg = 'You have successfully registered !'
            # return render_template('index.html.html', msg=msg)


    elif request.method == 'POST':

        msg = 'Please fill out the form !'
    return render_template('register.html',msg=msg)

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

@app.route('/addChannel', methods=['GET','POST'])
def add_channel():

    if 'loggedin' in session:

        conn = mysqldb.connect()
        cursor = conn.cursor()
        if request.method == 'POST' and 'appointment' in request.form:
            name = request.form['name']
            if(name==''):
                appointment = request.form['appointment']
                date,time,meridiem=appointment.split(' ')
                hour,min = (int(x) for x in time.split(':'))
                month,day, year = (int(x) for x in date.split('/'))
                ans = datetime.date(year, month, day)
                weekday=ans.strftime("%A")
                cursor.execute('SELECT dt.docterId AS docID, dt.day AS day, dt.timeStart AS start, dt.timeEnd AS end, d.docFirstName AS firstName, d.docLastName AS lastName FROM DoctorTimeSlots dt JOIN Doctor d ON dt.docterId = d.docId WHERE day = % s', weekday)
                doctorList = cursor.fetchall()
                return render_template("channelling.html", doctorList=doctorList)

            elif request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'mobNo' in request.form and 'appointment' in request.form and 'status' in request.form or 'message' in request.form:

                email = request.form['email']
                mobNo = request.form['mobNo']
                confirmAppointments = request.form['appointment']
                doctorId = request.form['doctorId']
                status = 'Pending'
                msg = request.form['message']
                confirmDate, confirmTime, confirmMeridiem = confirmAppointments.split(' ')
                sql = "INSERT INTO Chanelling(id,channel_date ,channel_time , docterId , status) VALUES(%s, %s, %s,%s, %s)"
                data = ((session['id']), confirmDate, confirmTime, doctorId, status)

                cursor.execute(sql, data)
                conn.commit()


        return render_template("channelling.html",)
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

#display selected doctor

@app.route('/doctorSearch')
def doctor_search():

    try:
        if 'loggedin' in session:
            conn = mysqldb.connect()
            cursor = conn.cursor()
            msg=''
            account=[]
            if request.method == 'GET'and 'email' in request.form:
                email = request.form['email']
                cursor.execute(
                    'SELECT docId ,docFirstName ,docLastName ,docAge ,docEmail , docGender ,docAddress , docMobileNo  FROM Doctor WHERE docEmail=%s',email)
                account = cursor.fetchone()
                if account is None:
                    msg='There is No Doctor Registered From This Email. ('+email+')'

            return render_template("admin.html", account=account,msg=msg)
        return redirect(url_for('login'))
    except Exception as e:
        print(e)
    finally:
        cursor.close()
        conn.close()



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
