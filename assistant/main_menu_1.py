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
        MainWindow.resize(1283, 676)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photoOutputWindow = QtWidgets.QLabel(self.centralwidget)
        self.photoOutputWindow.setGeometry(QtCore.QRect(20, 130, 591, 381))
        self.photoOutputWindow.setFrameShape(QtWidgets.QFrame.Box)
        self.photoOutputWindow.setText("")
        self.photoOutputWindow.setObjectName("photoOutputWindow")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 40, 591, 71))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.textCommand = QtWidgets.QLabel(self.layoutWidget)
        self.textCommand.setObjectName("textCommand")
        self.horizontalLayout.addWidget(self.textCommand)
        spacerItem1 = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.chooseSensorTypeBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.chooseSensorTypeBtn.setObjectName("chooseSensorTypeBtn")
        self.verticalLayout.addWidget(self.chooseSensorTypeBtn)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(619, 79, 591, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textInfo = QtWidgets.QLabel(self.layoutWidget1)
        self.textInfo.setObjectName("textInfo")
        self.horizontalLayout_2.addWidget(self.textInfo)
        self.textOutput = QtWidgets.QTextEdit(self.layoutWidget1)
        self.textOutput.setObjectName("textOutput")
        self.horizontalLayout_2.addWidget(self.textOutput)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.typeCorrectCBox = QtWidgets.QCheckBox(self.layoutWidget1)
        self.typeCorrectCBox.setObjectName("typeCorrectCBox")
        self.horizontalLayout_3.addWidget(self.typeCorrectCBox)
        self.typeSensor = QtWidgets.QPushButton(self.centralwidget)
        self.typeSensor.setGeometry(QtCore.QRect(740, 40, 361, 31))
        self.typeSensor.setObjectName("typeSensor")
        self.editedPicture = QtWidgets.QLabel(self.centralwidget)
        self.editedPicture.setGeometry(QtCore.QRect(620, 130, 591, 381))
        self.editedPicture.setFrameShape(QtWidgets.QFrame.Box)
        self.editedPicture.setText("")
        self.editedPicture.setObjectName("editedPicture")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1283, 23))
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
        self.textCommand.setText(_translate("MainWindow", "Установите деталь под считывающую камеру!"))
        self.chooseSensorTypeBtn.setText(_translate("MainWindow", "Включить камеру"))
        self.textInfo.setText(_translate("MainWindow", "Тип датчика:"))
        self.typeCorrectCBox.setText(_translate("MainWindow", "Определен"))
        self.typeSensor.setText(_translate("MainWindow", "Определить тип датчика"))
        self.menuResistor_Sticker_Assistant.setTitle(_translate("MainWindow", "Resistor Sticker Assistant"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
