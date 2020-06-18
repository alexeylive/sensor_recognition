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
        MainWindow.resize(705, 666)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photoOutputWindow = QtWidgets.QLabel(self.centralwidget)
        self.photoOutputWindow.setGeometry(QtCore.QRect(60, 200, 591, 381))
        self.photoOutputWindow.setFrameShape(QtWidgets.QFrame.Box)
        self.photoOutputWindow.setText("")
        self.photoOutputWindow.setObjectName("photoOutputWindow")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 60, 591, 71))
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
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(60, 140, 591, 41))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.typeSensor = QtWidgets.QPushButton(self.widget)
        self.typeSensor.setObjectName("typeSensor")
        self.horizontalLayout_4.addWidget(self.typeSensor)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textInfo = QtWidgets.QLabel(self.widget)
        self.textInfo.setObjectName("textInfo")
        self.horizontalLayout_2.addWidget(self.textInfo)
        self.textOutput = QtWidgets.QTextEdit(self.widget)
        self.textOutput.setObjectName("textOutput")
        self.horizontalLayout_2.addWidget(self.textOutput)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.typeCorrectCBox = QtWidgets.QCheckBox(self.widget)
        self.typeCorrectCBox.setObjectName("typeCorrectCBox")
        self.horizontalLayout_3.addWidget(self.typeCorrectCBox)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 705, 23))
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
        self.typeSensor.setText(_translate("MainWindow", "Определить тип датчика"))
        self.textInfo.setText(_translate("MainWindow", "Тип датчика:"))
        self.typeCorrectCBox.setText(_translate("MainWindow", "Определен"))
        self.menuResistor_Sticker_Assistant.setTitle(_translate("MainWindow", "Resistor Sticker Assistant"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
