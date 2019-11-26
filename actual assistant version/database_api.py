import argparse
import hashlib
import uuid

import pymysql.cursors


class AssistantApp:
    def __init__(self, db_pwd, db_name):
        self._salt = "5051"
        self._token_dict = ()
        self._tokens_dict = {}
        self.db_connection = pymysql.connect(host='localhost',
                                             user='ivan',
                                             password=db_pwd,
                                             db=db_name,
                                             charset='utf8mb4',
                                             cursorclass=pymysql.cursors.DictCursor)

    def __del__(self):
        self.db_connection.close()

    def get_hash(self, sens_data):
        return hashlib.sha256((self._salt + sens_data).encode('utf-8')).hexdigest()

    def add_new_worker(self, worker_name, worker_surname, worker_password):
        with self.db_connection.cursor() as cursor:
            sql = "SELECT WorkerID FROM Worker WHERE WorkerName=%s AND WorkerSurname=%s"
            cursor.execute(sql, (worker_name, worker_surname))
            if cursor.fetchone() is not None:
                raise ValueError(f'User name: "{worker_name}" and surname: "{worker_surname}" are already registered')
            sql = "INSERT INTO Worker (WorkerName, WorkerSurname, WorkerPassword) VALUES (%s, %s, %s)"
            cursor.execute(sql, (worker_name, worker_surname, self.get_hash(worker_password)))
        self.db_connection.commit()

    def authenticate_worker(self, worker_name, worker_surname, worker_password):
        with self.db_connection.cursor() as cursor:
            sql = "SELECT WorkerID FROM Worker WHERE WorkerName=%s AND WorkerSurname=%s AND WorkerPassword=%s"
            cursor.execute(sql, (worker_name, worker_surname, self.get_hash(worker_password)))
            worker_id_dict = cursor.fetchone()
        if worker_id_dict is None:
            raise ValueError('Incorrect Name/Password pair. Please, try again')
        worker_token = uuid.uuid4()
        user_id = worker_id_dict['WorkerID']
        self._tokens_dict[worker_token] = user_id
        return worker_token

    # TODO(@me): make that fucking type fields
    def add_sensor(self,
                   sensor_code: str,
                   sensor_rated_load: int,
                   sensor_weight: int,
                   sensor_accuracy_class: str,
                   sensor_material: str,
                   sensor_input_impedance: int,
                   sensor_output_impedance: int):
        with self.db_connection.cursor() as cursor:
            sql = "SELECT SensorID FROM Sensor WHERE SensorCode=%s"
            cursor.execute(sql, sensor_code)
            if cursor.fetchone() is not None:
                return f'Sensor with code: {sensor_code} is already exist', None
            sql = "INSERT INTO Sensor (SensorCode, \
                                       SensorRatedLoad, \
                                       SensorWeight, \
                                       SensorAccuracyClass, \
                                       SensorMaterial, \
                                       SensorInputImpedance, \
                                       SensorOutputImpedance) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (sensor_code,
                                 sensor_rated_load,
                                 sensor_weight,
                                 sensor_accuracy_class,
                                 sensor_material,
                                 sensor_input_impedance,
                                 sensor_output_impedance))
        self.db_connection.commit()
        return True, None

    def get_sensor_parameters(self, sensor_code):
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * from Sensor WHERE SensorCode=%s"
            cursor.execute(sql, sensor_code)
            sensor_parameters = cursor.fetchone()
        if sensor_parameters is None:
            return f"Sensor with code: {sensor_code} does not exist", None
        return None, sensor_parameters


if __name__ == "__main__":
    assistant = AssistantApp('', 'AssistantSensorsDatabase')
    # assistant.add_sensor('P1D4R4S', 420, 9000, 'A2', 'Steel', 1, 2)
    _, p = assistant.get_sensor_parameters('P1D4R4S')
    print(p)