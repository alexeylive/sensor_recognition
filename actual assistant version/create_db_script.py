import pymysql


def create_db():
    database_server_ip = "localhost"  # IP address of the MySQL database server
    database_username = "ivan"  # User name of the database server
    database_userpassword = ""  # Password for the database user
    database_name = "AssistantSensorsDatabase"  # Name of the database that is to be created
    char_set = "utf8mb4"  # Character set

    cursor = pymysql.cursors.DictCursor
    connection = pymysql.connect(host=database_server_ip,
                                 user=database_username,
                                 password=database_userpassword,
                                 charset=char_set,
                                 cursorclass=cursor)
    try:
        cursor = connection.cursor()
        sql = "CREATE DATABASE " + database_name
        cursor.execute(sql)
        connection.commit()

        sql = "USE " + database_name
        cursor.execute(sql)
        connection.commit()

        sql = "CREATE TABLE Sensor (SensorID INT NOT NULL PRIMARY KEY AUTO_INCREMENT, \
                                    SensorCode VARCHAR(256) NOT NULL, \
                                    SensorRatedLoad INT NOT NULL, \
                                    SensorWeight INT NOT NULL, \
                                    SensorAccuracyClass VARCHAR(256) NOT NULL, \
                                    SensorMaterial VARCHAR(256) NOT NULL, \
                                    SensorInputImpedance INT NOT NULL, \
                                    SensorOutputImpedance INT NOT NULL)"
        cursor.execute(sql)
        connection.commit()

        sql = "CREATE TABLE Worker (WorkerID INT NOT NULL PRIMARY KEY AUTO_INCREMENT, \
                                    WorkerName VARCHAR(256) NOT NULL, \
                                    WorkerSurname VARCHAR(256) NOT NULL, \
                                    WorkerPassword VARCHAR(1024) NOT NULL)"
        cursor.execute(sql)
        connection.commit()

    finally:
        connection.close()


if __name__ == '__main__':
    create_db()
