# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys
import cv2
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Ui_MainWindow(object):
    def __init__(self):
        self.video_recording_btn=None
        #self.label=None
        self.open_cam=None
        self.close_cam=None
        self.open_danger=None
        self.close_danger=None
        self.lcd_class_1=None
        self.lcd_class_2=None
        self.lcd_class_3=None
        self.lcd_class_4=None
        self.lcd_class_5=None
        self.lcd_class_6=None
        self.lcd_class_7=None
        self.lcd_class_8=None
        self.lcd_class_9=None
        self.lcd_class_10=None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1071, 680)
        MainWindow.setMinimumSize(QtCore.QSize(1071, 680))
        MainWindow.setMaximumSize(QtCore.QSize(1071, 680))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("/UI/image/github.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(180, 40, 871, 421))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setObjectName("frame")
        self.video_feed = QtWidgets.QLabel(self.frame)
        self.video_feed.setGeometry(QtCore.QRect(0, 0, 871, 421))
        self.video_feed.setStyleSheet("QLabel\n"
"{\n"
"  border:2px solid rgb(0, 0, 0);\n"
"}")
        self.video_feed.setText("")
        self.video_feed.setObjectName("video_feed")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 40, 171, 421))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_2.setObjectName("frame_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 220, 150, 131))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(20, 30, 121, 81))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setPixmap(QtGui.QPixmap(""))
        self.label.setScaledContents(True)
        self.groupBox_2 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 150, 191))
        self.groupBox_2.setObjectName("groupBox_2")
        self.open_cam = QtWidgets.QPushButton(self.groupBox_2)
        self.open_cam.setGeometry(QtCore.QRect(10, 20, 120, 41))
        self.open_cam.setObjectName("open_cam")
        self.open_danger = QtWidgets.QPushButton(self.groupBox_2)
        self.open_danger.setGeometry(QtCore.QRect(10, 80, 120, 41))
        self.open_danger.setObjectName("open_danger")
        self.close_danger = QtWidgets.QPushButton(self.groupBox_2)
        self.close_danger.setGeometry(QtCore.QRect(10, 138, 120, 41))
        self.close_danger.setObjectName("close_danger")
        self.close_cam = QtWidgets.QPushButton(self.frame_2)
        self.close_cam.setGeometry(QtCore.QRect(40, 370, 101, 31))
        self.close_cam.setCheckable(True)
        self.close_cam.setObjectName("close_cam")
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 460, 1041, 61))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_3.setObjectName("frame_3")
        self.lcd_class_5 = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd_class_5.setGeometry(QtCore.QRect(920, 22, 91, 21))
        self.lcd_class_5.setObjectName("lcd_class_5")
        self.label_class_5 = QtWidgets.QLabel(self.frame_3)
        self.label_class_5.setGeometry(QtCore.QRect(840, 20, 71, 21))
        self.label_class_5.setObjectName("label_class_5")
        self.label_class_2 = QtWidgets.QLabel(self.frame_3)
        self.label_class_2.setGeometry(QtCore.QRect(220, 20, 71, 21))
        self.label_class_2.setObjectName("label_class_2")
        self.label_class_1 = QtWidgets.QLabel(self.frame_3)
        self.label_class_1.setGeometry(QtCore.QRect(20, 20, 71, 21))
        self.label_class_1.setObjectName("label_class_1")
        self.lcd_class_1 = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd_class_1.setGeometry(QtCore.QRect(90, 22, 91, 21))
        self.lcd_class_1.setObjectName("lcd_class_1")
        self.lcd_class_2 = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd_class_2.setGeometry(QtCore.QRect(290, 22, 91, 21))
        self.lcd_class_2.setObjectName("lcd_class_2")
        self.llabel_class_3 = QtWidgets.QLabel(self.frame_3)
        self.llabel_class_3.setGeometry(QtCore.QRect(420, 20, 71, 21))
        self.llabel_class_3.setObjectName("llabel_class_3")
        self.lcd_class_3 = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd_class_3.setGeometry(QtCore.QRect(500, 22, 91, 21))
        self.lcd_class_3.setObjectName("lcd_class_3")
        self.label_class_4 = QtWidgets.QLabel(self.frame_3)
        self.label_class_4.setGeometry(QtCore.QRect(630, 20, 71, 21))
        self.label_class_4.setObjectName("label_class_4")
        self.lcd_class_4 = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd_class_4.setGeometry(QtCore.QRect(710, 22, 91, 21))
        self.lcd_class_4.setObjectName("lcd_class_4")
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(10, 580, 1041, 61))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_5.setObjectName("frame_5")
        self.exit_btn = QtWidgets.QPushButton(self.frame_5)
        self.exit_btn.setGeometry(QtCore.QRect(480, 20, 91, 31))
        self.exit_btn.setObjectName("exit_btn")
        self.Logo_label = QtWidgets.QLabel(self.frame_5)
        self.Logo_label.setGeometry(QtCore.QRect(870, 10, 54, 41))
        self.Logo_label.setInputMethodHints(QtCore.Qt.ImhNone)
        self.Logo_label.setText("")
        self.Logo_label.setPixmap(QtGui.QPixmap("UI/image/github.png"))
        self.Logo_label.setScaledContents(True)
        self.Logo_label.setObjectName("Logo_label")
        self.name_label = QtWidgets.QLabel(self.frame_5)
        self.name_label.setGeometry(QtCore.QRect(930, 20, 91, 20))
        self.name_label.setObjectName("name_label")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(10, 520, 1041, 61))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setObjectName("frame_4")
        self.lcd_class_10 = QtWidgets.QLCDNumber(self.frame_4)
        self.lcd_class_10.setGeometry(QtCore.QRect(920, 22, 91, 21))
        self.lcd_class_10.setObjectName("lcd_class_10")
        self.label_class_10 = QtWidgets.QLabel(self.frame_4)
        self.label_class_10.setGeometry(QtCore.QRect(840, 20, 71, 21))
        self.label_class_10.setObjectName("label_class_10")
        self.label_class_7 = QtWidgets.QLabel(self.frame_4)
        self.label_class_7.setGeometry(QtCore.QRect(220, 20, 71, 21))
        self.label_class_7.setObjectName("label_class_7")
        self.label_class_6 = QtWidgets.QLabel(self.frame_4)
        self.label_class_6.setGeometry(QtCore.QRect(20, 20, 71, 21))
        self.label_class_6.setObjectName("label_class_6")
        self.lcd_class_6 = QtWidgets.QLCDNumber(self.frame_4)
        self.lcd_class_6.setGeometry(QtCore.QRect(90, 22, 91, 21))
        self.lcd_class_6.setObjectName("lcd_class_6")
        self.lcd_class_7 = QtWidgets.QLCDNumber(self.frame_4)
        self.lcd_class_7.setGeometry(QtCore.QRect(290, 22, 91, 21))
        self.lcd_class_7.setObjectName("lcd_class_7")
        self.label_class_8 = QtWidgets.QLabel(self.frame_4)
        self.label_class_8.setGeometry(QtCore.QRect(420, 20, 71, 21))
        self.label_class_8.setObjectName("label_class_8")
        self.lcd_class_8 = QtWidgets.QLCDNumber(self.frame_4)
        self.lcd_class_8.setGeometry(QtCore.QRect(500, 22, 91, 21))
        self.lcd_class_8.setObjectName("lcd_class_8")
        self.label_class_9 = QtWidgets.QLabel(self.frame_4)
        self.label_class_9.setGeometry(QtCore.QRect(630, 20, 71, 21))
        self.label_class_9.setObjectName("label_class_9")
        self.lcd_class_9 = QtWidgets.QLCDNumber(self.frame_4)
        self.lcd_class_9.setGeometry(QtCore.QRect(710, 22, 91, 21))
        self.lcd_class_9.setObjectName("lcd_class_9")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menu_bar = QtWidgets.QMenuBar(MainWindow)
        self.menu_bar.setGeometry(QtCore.QRect(0, 0, 1071, 26))
        self.menu_bar.setDefaultUp(False)
        self.menu_bar.setObjectName("menu_bar")
        MainWindow.setMenuBar(self.menu_bar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "异常驾驶行为识别系统"))
        self.groupBox_3.setTitle(_translate("MainWindow", "预警"))
        self.groupBox_2.setTitle(_translate("MainWindow", "功能栏"))
        self.open_cam.setText(_translate("MainWindow", "进行识别"))
        self.open_danger.setText(_translate("MainWindow", "开启预警"))
        self.close_danger.setText(_translate("MainWindow", "关闭预警"))
        self.close_cam.setText(_translate("MainWindow", "关闭摄像头"))
        self.label_class_5.setText(_translate("MainWindow", "左手打电话"))
        self.label_class_2.setText(_translate("MainWindow", "右手打字"))
        self.label_class_1.setText(_translate("MainWindow", "安全驾驶"))
        self.llabel_class_3.setText(_translate("MainWindow", "右手打电话"))
        self.label_class_4.setText(_translate("MainWindow", "左手打字"))
        self.exit_btn.setText(_translate("MainWindow", "Exit"))
        self.name_label.setText(_translate("MainWindow", "imeGao伯爵"))
        self.label_class_10.setText(_translate("MainWindow", "和乘客交流"))
        self.label_class_7.setText(_translate("MainWindow", "拿饮料喝"))
        self.label_class_6.setText(_translate("MainWindow", "调收音机"))
        self.label_class_8.setText(_translate("MainWindow", "拿后面东西"))
        self.label_class_9.setText(_translate("MainWindow", "打理头部"))

