from application import app
from flaskext.mysql import MySQL

mysqldb = MySQL()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'chamma'
app.config['MYSQL_DATABASE_PASSWORD'] = 'skinsafe'
app.config['MYSQL_DATABASE_DB'] = 'SkinSafe'
app.config['MYSQL_DATABASE_HOST'] = 'database-skinsafe.cfjzzf8ivqho.us-east-1.rds.amazonaws.com'



mysqldb.init_app(app)