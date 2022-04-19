from app import app
from flaskext.mysql import MySQL

mysqldb = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'SkinSafe'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'



mysqldb.init_app(app)