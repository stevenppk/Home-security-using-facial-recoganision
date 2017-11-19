# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import os

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Bell(object):
    
    def bellclick(self):
	os.system("./imgdetection.sh")

    def setupUi(self, Bell):
        Bell.setObjectName(_fromUtf8("Bell"))
        Bell.resize(390, 415)
        self.centralwidget = QtGui.QWidget(Bell)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(-110, -110, 581, 531))
        self.pushButton.setStyleSheet(_fromUtf8("\n"
"background-image: url(:/newPrefix/media/bell.png);"))
        self.pushButton.setText(_fromUtf8(""))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
	#
	self.pushButton.clicked.connect(self.bellclick)
	#
        Bell.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(Bell)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Bell.setStatusBar(self.statusbar)

        self.retranslateUi(Bell)
        QtCore.QMetaObject.connectSlotsByName(Bell)

    def retranslateUi(self, Bell):
        Bell.setWindowTitle(_translate("Bell", "MainWindow", None))

import resources_rc

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Bell = QtGui.QMainWindow()
    ui = Ui_Bell()
    ui.setupUi(Bell)
    Bell.show()
    sys.exit(app.exec_())

