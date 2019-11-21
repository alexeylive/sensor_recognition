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


from main_menu_2 import *
import sensor_recognition


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
        self.ui.spotTypeSensor.clicked.connect(self.get_camera_frame)

        label = QtGui.QPixmap("15646708886980.jpg")
        self.ui.videoStreamWindow.setPixmap(label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

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
            self.ui.sensorInfo.setText('Сначала включите камеру')

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
