import sys
import queue
import threading
import time
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from gui_3 import *
import sensor_recognition
import digital_twin_class

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
        self.setWindowTitle("Контроль точности")

        self.ui.enableCameraButton.clicked.connect(self.start_camera_stream)
        self.ui.getCameraFrameButton.clicked.connect(self.get_camera_frame)
        self.ui.spotTypeSensorButton.clicked.connect(self.define_sensor_parameters)
        self.ui.numberOfResistor.valueChanged.connect(self.set_resistor_number_from_spinbox)
        self.ui.accuracyButton.clicked.connect(self.show_sensor_accuracy)

        label = QtGui.QPixmap("video.png")
        self.ui.videoStreamWindow.setAlignment(Qt.AlignCenter)
        self.ui.videoStreamWindow.setPixmap(label)
        avatar = QtGui.QPixmap("user_avatar.png")
        self.ui.avatarWorkerPicture.setAlignment(Qt.AlignCenter)
        self.ui.avatarWorkerPicture.setPixmap(avatar)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.resistors = {}

    def define_sensor_parameters(self):
        sensor_id = self.ui.sensorInfoInput.toPlainText()
        if sensor_id == '':
            self.ui.sensorTypePicture.setText('Введите идентификатор датчика')
            return

        # go to DB, return parameters of current sensor
        # sensor_parameters = get_sensor_parameters(id)
        # sensor_name from db

        sensor_rated_load = 30
        sensor_weight = 8
        sensor_accuracy_class = 'С1'
        sensor_material = 'Сталь'
        sensor_input_impedance = 750
        sensor_output_impedance = 700

        self.ui.sensorRatedLoad.setText(str(sensor_rated_load))
        self.ui.sensorWeight.setText(str(sensor_weight))
        self.ui.sensorAccuracyClass.setText(str(sensor_accuracy_class))
        self.ui.sensorMaterial.setText(str(sensor_material))
        self.ui.sensorInputImpedance.setText(str(sensor_input_impedance))
        self.ui.sensorOutputImpedances.setText(str(sensor_output_impedance))

        sensor_type_pic = QtGui.QPixmap("sensor_type.jpg")
        self.ui.sensorTypePicture.setAlignment(Qt.AlignCenter)
        self.ui.sensorTypePicture.setPixmap(sensor_type_pic.scaled(591, 371, QtCore.Qt.KeepAspectRatio))

    def get_camera_frame(self):
        if capturing:
            frame = frames_queue.get()
            # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            recognized_frame = sensor_recognition.get_image_with_realogram(frame)
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

    def set_resistor_number_from_spinbox(self):
        self.resistors.update({self.ui.numberOfResistor.value(): (10, 30, 10, 30)})

    def start_camera_stream(self):
        """
        Activating camera_thread thread on a %button_name% pushing.
        """
        global capturing
        capturing = True
        camera_thread.start()

        self.ui.enableCameraButton.setEnabled(False)
        self.ui.enableCameraButton.setText('Камера включается...')

    def show_sensor_accuracy(self):
        X, Y, phi = [9.9, 30.1, 10.3, 29], [0, 0, 0, 0], [np.pi/100, -np.pi/100, np.pi-np.pi/100, np.pi]
        sensor_info = (X, Y, phi)
        model = digital_twin_class.DigitalTwin(sensor_info)
        result = model.get_sensor_result()
        print(result)
        # if 0 <= result <= 5:
        #     self.ui.accuracyValue.setText('Датчик исправен')
        if 0 <= result <= 5:
            self.ui.accuracyValue.setText('Брак. Перемонтаж 1 3 резисторов')


if __name__ == '__main__':
    camera_thread = threading.Thread(target=camera_saving)

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.png"))
    my_app = MyWin()
    my_app.show()

    sys.exit(app.exec_())
