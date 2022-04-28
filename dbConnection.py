from application import app
from flaskext.mysql import MySQL

mysqldb = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'admin'
app.config['MYSQL_DATABASE_PASSWORD'] = 'MC12345678'
app.config['MYSQL_DATABASE_DB'] = 'database-1'
app.config['MYSQL_DATABASE_HOST'] = 'database-1.cfjzzf8ivqho.us-east-1.rds.amazonaws.com'



mysqldb.init_app(app)