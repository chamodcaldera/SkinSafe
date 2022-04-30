from application import app
from flaskext.mysql import MySQL

mysqldb = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'admin'
app.config['MYSQL_DATABASE_PASSWORD'] = 'fOHh2cA0'
app.config['MYSQL_DATABASE_DB'] = 'skinsafe'
app.config['MYSQL_DATABASE_HOST'] = 'mysql-77609-0.cloudclusters.net'



mysqldb.init_app(app)