# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'assistant_screen.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1229, 668)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.videoStreamWindow = QtWidgets.QLabel(self.centralwidget)
        self.videoStreamWindow.setGeometry(QtCore.QRect(20, 130, 591, 381))
        self.videoStreamWindow.setFrameShape(QtWidgets.QFrame.Box)
        self.videoStreamWindow.setText("")
        self.videoStreamWindow.setObjectName("videoStreamWindow")
        self.spotTypeSensor = QtWidgets.QPushButton(self.centralwidget)
        self.spotTypeSensor.setGeometry(QtCore.QRect(320, 520, 591, 41))
        self.spotTypeSensor.setObjectName("spotTypeSensor")
        self.editedPictureWindow = QtWidgets.QLabel(self.centralwidget)
        self.editedPictureWindow.setGeometry(QtCore.QRect(620, 130, 591, 381))
        self.editedPictureWindow.setFrameShape(QtWidgets.QFrame.Box)
        self.editedPictureWindow.setText("")
        self.editedPictureWindow.setObjectName("editedPictureWindow")
        self.enableCameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.enableCameraButton.setGeometry(QtCore.QRect(320, 60, 591, 51))
        self.enableCameraButton.setObjectName("enableCameraButton")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(320, 32, 591, 22))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.prepareDetailCommand = QtWidgets.QLabel(self.widget)
        self.prepareDetailCommand.setObjectName("prepareDetailCommand")
        self.horizontalLayout.addWidget(self.prepareDetailCommand)
        spacerItem1 = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(321, 571, 591, 41))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.sensorInfo = QtWidgets.QLabel(self.widget1)
        self.sensorInfo.setObjectName("sensorInfo")
        self.horizontalLayout_2.addWidget(self.sensorInfo)
        self.sensorInfoOutput = QtWidgets.QTextEdit(self.widget1)
        self.sensorInfoOutput.setObjectName("sensorInfoOutput")
        self.horizontalLayout_2.addWidget(self.sensorInfoOutput)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1229, 23))
        self.menubar.setObjectName("menubar")
        self.menuResistor_Sticker_Assistant = QtWidgets.QMenu(self.menubar)
        self.menuResistor_Sticker_Assistant.setObjectName("menuResistor_Sticker_Assistant")
        MainWindow.setMenuBar(self.menubar)
        self.menuResistor_Sticker_Assistant.addSeparator()
        self.menubar.addAction(self.menuResistor_Sticker_Assistant.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.spotTypeSensor.setText(_translate("MainWindow", "Определить тип датчика"))
        self.enableCameraButton.setText(_translate("MainWindow", "Включить камеру"))
        self.prepareDetailCommand.setText(_translate("MainWindow", "Установите деталь под считывающую камеру!"))
        self.sensorInfo.setText(_translate("MainWindow", "Тип датчика:"))
        self.menuResistor_Sticker_Assistant.setTitle(_translate("MainWindow", "Resistor Sticker Assistant"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
