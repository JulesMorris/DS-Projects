username = "jemison_1743"
password =  "zYETFUTBw3p7DV7oo1E5GjpqkKN8X4q7"
host = "157.230.209.171"

def get_db_url(database, host=host, username=username, password=password):
	return f'mysql+pymysql://{username}:{password}@{host}/{database}'
