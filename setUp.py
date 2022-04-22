import mysql.connector #Importing Connector package



def connection():
    mysqldb=mysql.connector.connect(host="localhost",user="root",password="")#established connection
    mycursor=mysqldb.cursor()#cursor() method create a cursor object
    mycursor.execute("CREATE DATABASE IF NOT EXISTS SkinSafe")  # Execute SQL Query to create a database
    # mycursor = mysqldb.cursor()  # cursor() method create a cursor object
    mycursor.execute("USE SkinSafe")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS User (id INT NOT NULL AUTO_INCREMENT,firstName VARCHAR (50) NOT NULL ,lastName VARCHAR (50) NOT NULL ,age VARCHAR (3) NOT NULL,email VARCHAR(50) NOT NULL,password VARCHAR (255) NOT NULL, gender VARCHAR (10)NOT NULL,address VARCHAR (255), mobileNo VARCHAR(10),PRIMARY KEY (id))")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS Doctor (docId INT NOT NULL AUTO_INCREMENT,docFirstName VARCHAR (50) NOT NULL ,docLastName VARCHAR (50) NOT NULL ,docAge VARCHAR (3) NOT NULL,docEmail VARCHAR(255) NOT NULL,docPassword VARCHAR (255) NOT NULL, docGender VARCHAR (10)NOT NULL,docAddress VARCHAR (255), docMobileNo VARCHAR(10),PRIMARY KEY (docId))")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS Prescription (recordId INT NOT NULL AUTO_INCREMENT, id INT NOT NULL,prescription LONGBLOB ,PRIMARY KEY (recordId), FOREIGN KEY (id)REFERENCES User (id)ON DELETE CASCADE)")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS TestReports (recordId INT NOT NULL AUTO_INCREMENT, id INT NOT NULL,testReports LONGBLOB ,PRIMARY KEY (recordId), FOREIGN KEY (id)REFERENCES User (id)ON DELETE CASCADE)")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS Channelling (channelId INT NOT NULL AUTO_INCREMENT, id INT NOT NULL,channel_date VARCHAR (10) NOT NULL,channel_time  VARCHAR (10)  NOT NULL , doctorId INT NOT NULL, status VARCHAR(10),message VARCHAR(100), PRIMARY KEY (channelId), FOREIGN KEY (id)REFERENCES User (id) ON DELETE CASCADE,FOREIGN KEY (doctorId)REFERENCES Doctor (docId) ON DELETE CASCADE)")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS Admin (id INT NOT NULL AUTO_INCREMENT,firstName VARCHAR (50) NOT NULL ,lastName VARCHAR (50) NOT NULL ,email VARCHAR(50) NOT NULL,password VARCHAR (255) NOT NULL, mobileNo VARCHAR(10),PRIMARY KEY (id))")  # Execute SQL Query to create a table into your database
    mycursor.execute("CREATE TABLE IF NOT EXISTS DoctorTimeSlots (timeSlotId INT NOT NULL AUTO_INCREMENT,doctorId INT NOT NULL  ,day VARCHAR (20) NOT NULL ,timeStart VARCHAR(10) NOT NULL, timeEnd VARCHAR(10) NOT NULL, PRIMARY KEY (timeSlotId),FOREIGN KEY (doctorId)REFERENCES doctor (docId)ON DELETE CASCADE)")  # Execute SQL Query to create a table into your database

    # mysqldb.close()  # Connection Close
    mysqldb.close()







