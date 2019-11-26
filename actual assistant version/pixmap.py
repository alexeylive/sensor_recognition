import sys
import queue
import threading
import time
import os

import cv2
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor


from gui import *
import sensor_recognition
import database_api

capturing = False
frames_queue = queue.Queue()


def camera_saving():
    video = cv2.VideoCapture(0)

    while capturing:
        check, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frames_queue.qsize() < 10:
            frames_queue.put(frame)
        else:
            time.sleep(0.01)

    video.release()
    cv2.destroyAllWindows()


class MyWin(QtWidgets.QMainWindow):
    """
    Class for main screen of assistant helper program.
    """
    def __init__(self, parent=None):
        """
        Initialization of UI, starting QTimer.
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.enableCameraButton.clicked.connect(self.start_camera_stream)
        self.ui.getCameraFrameButton.clicked.connect(self.get_camera_frame)
        self.ui.spotTypeSensorButton.clicked.connect(self.define_sensor_parameters)

        label = QtGui.QPixmap("15646708886980.jpg")
        self.ui.videoStreamWindow.setPixmap(label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def set_empty_sensor_values(self):
        self.ui.sensorRatedLoad.setText('')
        self.ui.sensorWeight.setText('')
        self.ui.sensorAccuracyClass.setText('')
        self.ui.sensorMaterial.setText('')
        self.ui.sensorInputImpedance.setText('')
        self.ui.sensorOutputImpedances.setText('')

    def define_sensor_parameters(self):
        self.set_empty_sensor_values()

        sensor_id = self.ui.sensorInfoInput.toPlainText()
        if sensor_id == '':
            self.ui.sensorTypePicture.setText('Введите идентификатор датчика')
            return

        sensor_db = database_api.AssistantApp('', 'AssistantSensorsDatabase')
        err, sensor_parameters = sensor_db.get_sensor_parameters(sensor_id)
        if err:
            self.ui.sensorTypePicture.setText(err)
            return
        else:
            sensor_rated_load = sensor_parameters['SensorRatedLoad']
            sensor_weight = sensor_parameters['SensorWeight']
            sensor_accuracy_class = sensor_parameters['SensorAccuracyClass']
            sensor_material = sensor_parameters['SensorMaterial']
            sensor_input_impedance = sensor_parameters['SensorInputImpedance']
            sensor_output_impedance = sensor_parameters['SensorOutputImpedance']

        self.ui.sensorRatedLoad.setText(str(sensor_rated_load))
        self.ui.sensorWeight.setText(str(sensor_weight))
        self.ui.sensorAccuracyClass.setText(str(sensor_accuracy_class))
        self.ui.sensorMaterial.setText(str(sensor_material))
        self.ui.sensorInputImpedance.setText(str(sensor_input_impedance))
        self.ui.sensorOutputImpedances.setText(str(sensor_output_impedance))

        sensor_type_pic = QtGui.QPixmap("sensor_type_1.jpg")
        self.ui.sensorTypePicture.setPixmap(sensor_type_pic.scaled(591, 371, QtCore.Qt.KeepAspectRatio))

    def get_camera_frame(self):
        if capturing:
            frame = frames_queue.get()
            # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            recognized_frame, _ = sensor_recognition.find_displacement_of_centers(frame)
            if type(recognized_frame) is not int:
                # recognized_frame = cv2.cvtColor(recognized_frame, cv2.COLOR_GRAY2RGB)
                height, width, channel = recognized_frame.shape
                bytes_in_line = 3 * width
                qt_image = QtGui.QImage(recognized_frame.data, width, height, bytes_in_line, QtGui.QImage.Format_RGB888)
                qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
                self.ui.editedPictureWindow.setPixmap(qt_pixmap)

        else:
            self.ui.editedPictureWindow.setText('Сначала включите камеру')

    def update_frame(self):
        if not frames_queue.empty():
            self.ui.enableCameraButton.setText('Камера включена')
            frame = frames_queue.get()
            height, width, channel = frame.shape
            bytes_in_line = 3 * width
            qt_image = QtGui.QImage(frame.data, width, height, bytes_in_line, QtGui.QImage.Format_RGB888)
            qt_pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.ui.videoStreamWindow.setPixmap(qt_pixmap)

    def start_camera_stream(self):
        """
        Activating camera_thread thread on a %button_name% pushing.
        """
        global capturing
        capturing = True
        camera_thread.start()

        self.ui.enableCameraButton.setEnabled(False)
        self.ui.enableCameraButton.setText('Камера включается...')


if __name__ == '__main__':
    camera_thread = threading.Thread(target=camera_saving)

    app = QtWidgets.QApplication(sys.argv)
    my_app = MyWin()
    my_app.show()

    sys.exit(app.exec_())
