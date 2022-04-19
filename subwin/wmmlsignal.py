from PyQt5 import QtCore, QtGui, QtWidgets


# noinspection PyUnresolvedReferences
class Ui_WMML_signal(object):
    def setupUi(self, WMML_signal):
        WMML_signal.setObjectName("WMML_signal")
        WMML_signal.resize(279, 231)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(WMML_signal)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(WMML_signal)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(250, 150))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.okbtn = QtWidgets.QPushButton(WMML_signal)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.okbtn.setFont(font)
        self.okbtn.setObjectName("okbtn")
        self.horizontalLayout.addWidget(self.okbtn)
        self.cclbtn = QtWidgets.QPushButton(WMML_signal)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.cclbtn.setFont(font)
        self.cclbtn.setObjectName("cclbtn")
        self.horizontalLayout.addWidget(self.cclbtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(WMML_signal)
        self.cclbtn.clicked['bool'].connect(WMML_signal.close)
        QtCore.QMetaObject.connectSlotsByName(WMML_signal)

    def retranslateUi(self, WMML_signal):
        _translate = QtCore.QCoreApplication.translate
        WMML_signal.setWindowTitle(_translate("WMML_signal", "WMMLsingal"))
        self.groupBox.setTitle(_translate("WMML_signal", "Receivers"))
        self.okbtn.setText(_translate("WMML_signal", "OK"))
        self.cclbtn.setText(_translate("WMML_signal", "Cancel"))
