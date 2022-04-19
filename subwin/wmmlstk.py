from PyQt5 import QtCore, QtGui, QtWidgets


# noinspection PyUnresolvedReferences
class Ui_WMML_stk(object):
    def setupUi(self, WMML_stk):
        WMML_stk.setObjectName("WMML_stk")
        WMML_stk.resize(730, 660)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WMML_stk.sizePolicy().hasHeightForWidth())
        WMML_stk.setSizePolicy(sizePolicy)
        WMML_stk.setMinimumSize(QtCore.QSize(730, 660))
        WMML_stk.setMaximumSize(QtCore.QSize(730, 691))
        self.gridLayout_6 = QtWidgets.QGridLayout(WMML_stk)
        self.gridLayout_6.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.groupBox = QtWidgets.QGroupBox(WMML_stk)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(489, 612))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setMinimumSize(QtCore.QSize(461, 249))
        self.groupBox_4.setMaximumSize(QtCore.QSize(461, 249))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.b_load1 = QtWidgets.QPushButton(self.groupBox_4)
        self.b_load1.setMinimumSize(QtCore.QSize(117, 28))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.b_load1.setFont(font)
        self.b_load1.setIconSize(QtCore.QSize(40, 40))
        self.b_load1.setFlat(False)
        self.b_load1.setObjectName("b_load1")
        self.verticalLayout_4.addWidget(self.b_load1)
        self.addButton = QtWidgets.QPushButton(self.groupBox_4)
        self.addButton.setMinimumSize(QtCore.QSize(117, 43))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.addButton.setFont(font)
        self.addButton.setIconSize(QtCore.QSize(40, 40))
        self.addButton.setFlat(False)
        self.addButton.setObjectName("addButton")
        self.verticalLayout_4.addWidget(self.addButton)
        self.rmvButton = QtWidgets.QPushButton(self.groupBox_4)
        self.rmvButton.setMinimumSize(QtCore.QSize(117, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.rmvButton.setFont(font)
        self.rmvButton.setIconSize(QtCore.QSize(40, 40))
        self.rmvButton.setFlat(False)
        self.rmvButton.setObjectName("rmvButton")
        self.verticalLayout_4.addWidget(self.rmvButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_22 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.verticalLayout.addWidget(self.label_22)
        self.b_load_info = QtWidgets.QTextBrowser(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_load_info.sizePolicy().hasHeightForWidth())
        self.b_load_info.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.b_load_info.setFont(font)
        self.b_load_info.setObjectName("b_load_info")
        self.verticalLayout.addWidget(self.b_load_info)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_23 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_2.addWidget(self.label_23)
        self.b_load_b1 = QtWidgets.QTextBrowser(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_load_b1.sizePolicy().hasHeightForWidth())
        self.b_load_b1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.b_load_b1.setFont(font)
        self.b_load_b1.setObjectName("b_load_b1")
        self.verticalLayout_2.addWidget(self.b_load_b1)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_24 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_3.addWidget(self.label_24)
        self.b_load_b1_2 = QtWidgets.QTextBrowser(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_load_b1_2.sizePolicy().hasHeightForWidth())
        self.b_load_b1_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.b_load_b1_2.setFont(font)
        self.b_load_b1_2.setObjectName("b_load_b1_2")
        self.verticalLayout_3.addWidget(self.b_load_b1_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 3)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_11.addWidget(self.groupBox_4)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_9.sizePolicy().hasHeightForWidth())
        self.groupBox_9.setSizePolicy(sizePolicy)
        self.groupBox_9.setMinimumSize(QtCore.QSize(461, 248))
        self.groupBox_9.setMaximumSize(QtCore.QSize(461, 248))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_9.setFont(font)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.f_load = QtWidgets.QPushButton(self.groupBox_9)
        self.f_load.setMinimumSize(QtCore.QSize(117, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.f_load.setFont(font)
        self.f_load.setIconSize(QtCore.QSize(40, 40))
        self.f_load.setFlat(False)
        self.f_load.setObjectName("f_load")
        self.verticalLayout_10.addWidget(self.f_load)
        self.addButton_2 = QtWidgets.QPushButton(self.groupBox_9)
        self.addButton_2.setMinimumSize(QtCore.QSize(117, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.addButton_2.setFont(font)
        self.addButton_2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Graphics/rand.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.addButton_2.setIcon(icon)
        self.addButton_2.setIconSize(QtCore.QSize(40, 40))
        self.addButton_2.setFlat(False)
        self.addButton_2.setObjectName("addButton_2")
        self.verticalLayout_10.addWidget(self.addButton_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem1)
        self.horizontalLayout_4.addLayout(self.verticalLayout_10)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_25 = QtWidgets.QLabel(self.groupBox_9)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_25.setFont(font)
        self.label_25.setObjectName("label_25")
        self.verticalLayout_6.addWidget(self.label_25)
        self.f_load_info = QtWidgets.QTextBrowser(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.f_load_info.sizePolicy().hasHeightForWidth())
        self.f_load_info.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.f_load_info.setFont(font)
        self.f_load_info.setObjectName("f_load_info")
        self.verticalLayout_6.addWidget(self.f_load_info)
        self.horizontalLayout.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_26 = QtWidgets.QLabel(self.groupBox_9)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.verticalLayout_7.addWidget(self.label_26)
        self.f_load_f1 = QtWidgets.QTextBrowser(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.f_load_f1.sizePolicy().hasHeightForWidth())
        self.f_load_f1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.f_load_f1.setFont(font)
        self.f_load_f1.setObjectName("f_load_f1")
        self.verticalLayout_7.addWidget(self.f_load_f1)
        self.horizontalLayout.addLayout(self.verticalLayout_7)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_9.addLayout(self.horizontalLayout)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_20 = QtWidgets.QLabel(self.groupBox_9)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_8.addWidget(self.label_20)
        self.f_load_f1_2 = QtWidgets.QTextBrowser(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.f_load_f1_2.sizePolicy().hasHeightForWidth())
        self.f_load_f1_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.f_load_f1_2.setFont(font)
        self.f_load_f1_2.setObjectName("f_load_f1_2")
        self.verticalLayout_8.addWidget(self.f_load_f1_2)
        self.verticalLayout_9.addLayout(self.verticalLayout_8)
        self.horizontalLayout_4.addLayout(self.verticalLayout_9)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 3)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.verticalLayout_11.addWidget(self.groupBox_9)
        self.verticalLayout_11.setStretch(0, 3)
        self.verticalLayout_11.setStretch(1, 2)
        self.verticalLayout_13.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.ps_cv = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ps_cv.sizePolicy().hasHeightForWidth())
        self.ps_cv.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_cv.setFont(font)
        self.ps_cv.setText("")
        self.ps_cv.setObjectName("ps_cv")
        self.horizontalLayout_5.addWidget(self.ps_cv)
        self.verticalLayout_12.addLayout(self.horizontalLayout_5)
        self.ps_ps = QtWidgets.QCheckBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ps_ps.sizePolicy().hasHeightForWidth())
        self.ps_ps.setSizePolicy(sizePolicy)
        self.ps_ps.setMinimumSize(QtCore.QSize(350, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_ps.setFont(font)
        self.ps_ps.setChecked(False)
        self.ps_ps.setObjectName("ps_ps")
        self.verticalLayout_12.addWidget(self.ps_ps)
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.gridLayout_3.addLayout(self.verticalLayout_13, 0, 0, 1, 1)
        self.horizontalLayout_6.addWidget(self.groupBox)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.groupBox_3 = QtWidgets.QGroupBox(WMML_stk)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setMinimumSize(QtCore.QSize(208, 439))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_3.setFont(font)
        self.groupBox_3.setCheckable(True)
        self.groupBox_3.setChecked(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.hss_fd = QtWidgets.QComboBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_fd.setFont(font)
        self.hss_fd.setObjectName("hss_fd")
        self.hss_fd.addItem("")
        self.hss_fd.addItem("")
        self.hss_fd.addItem("")
        self.hss_fd.addItem("")
        self.hss_fd.addItem("")
        self.gridLayout_4.addWidget(self.hss_fd, 1, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.hss_times = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_times.setFont(font)
        self.hss_times.setObjectName("hss_times")
        self.gridLayout_4.addWidget(self.hss_times, 0, 1, 1, 1)
        self.verticalLayout_15.addLayout(self.gridLayout_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_5.setFont(font)
        self.groupBox_5.setCheckable(False)
        self.groupBox_5.setChecked(False)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.ms_tpeButton = QtWidgets.QRadioButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_tpeButton.setFont(font)
        self.ms_tpeButton.setChecked(True)
        self.ms_tpeButton.setObjectName("ms_tpeButton")
        self.verticalLayout_14.addWidget(self.ms_tpeButton)
        self.ms_rsaButton = QtWidgets.QRadioButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_rsaButton.setFont(font)
        self.ms_rsaButton.setObjectName("ms_rsaButton")
        self.verticalLayout_14.addWidget(self.ms_rsaButton)
        self.ms_saButton = QtWidgets.QRadioButton(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_saButton.setFont(font)
        self.ms_saButton.setObjectName("ms_saButton")
        self.verticalLayout_14.addWidget(self.ms_saButton)
        self.verticalLayout_15.addWidget(self.groupBox_5)
        self.verticalLayout_17.addLayout(self.verticalLayout_15)
        self.verticalLayout_18.addWidget(self.groupBox_2)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_2 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_16.addWidget(self.label_2)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 1, 0, 1, 1)
        self.hss_cv_min = QtWidgets.QLineEdit(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_cv_min.setFont(font)
        self.hss_cv_min.setObjectName("hss_cv_min")
        self.gridLayout_5.addWidget(self.hss_cv_min, 0, 1, 1, 1)
        self.hss_cv_max = QtWidgets.QLineEdit(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_cv_max.setFont(font)
        self.hss_cv_max.setObjectName("hss_cv_max")
        self.gridLayout_5.addWidget(self.hss_cv_max, 1, 1, 1, 1)
        self.verticalLayout_16.addLayout(self.gridLayout_5)
        self.hss_ps = QtWidgets.QCheckBox(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_ps.setFont(font)
        self.hss_ps.setChecked(False)
        self.hss_ps.setObjectName("hss_ps")
        self.verticalLayout_16.addWidget(self.hss_ps)
        self.verticalLayout_18.addWidget(self.groupBox_6)
        self.verticalLayout_19.addWidget(self.groupBox_3)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_19.addItem(spacerItem2)
        self.jzsdf = QtWidgets.QPushButton(WMML_stk)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.jzsdf.sizePolicy().hasHeightForWidth())
        self.jzsdf.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.jzsdf.setFont(font)
        self.jzsdf.setIconSize(QtCore.QSize(42, 42))
        self.jzsdf.setFlat(False)
        self.jzsdf.setObjectName("jzsdf")
        self.verticalLayout_19.addWidget(self.jzsdf)
        self.horizontalLayout_6.addLayout(self.verticalLayout_19)
        self.horizontalLayout_6.setStretch(0, 3)
        self.horizontalLayout_6.setStretch(1, 1)
        self.gridLayout_6.addLayout(self.horizontalLayout_6, 0, 1, 1, 1)
        self.infobox = QtWidgets.QLabel(WMML_stk)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.infobox.setFont(font)
        self.infobox.setObjectName("infobox")
        self.gridLayout_6.addWidget(self.infobox, 1, 1, 1, 1)

        self.retranslateUi(WMML_stk)
        QtCore.QMetaObject.connectSlotsByName(WMML_stk)

    def retranslateUi(self, WMML_stk):
        _translate = QtCore.QCoreApplication.translate
        WMML_stk.setWindowTitle(_translate("WMML_stk", "WMMLstk"))
        self.groupBox.setTitle(_translate("WMML_stk", "Model Parameters"))
        self.groupBox_4.setTitle(_translate("WMML_stk", "Base Estimator"))
        self.b_load1.setText(_translate("WMML_stk", "Load Model"))
        self.addButton.setText(_translate("WMML_stk", "Add Model To\n"
" Model List"))
        self.rmvButton.setText(_translate("WMML_stk", "Remove Model\n"
" From Model List"))
        self.label_22.setText(_translate("WMML_stk", "Loaded Model"))
        self.label_23.setText(_translate("WMML_stk", "Meta Model"))
        self.label_24.setText(_translate("WMML_stk", "Base Estimator List"))
        self.groupBox_9.setTitle(_translate("WMML_stk", "Final Estimator"))
        self.f_load.setText(_translate("WMML_stk", "Load Model"))
        self.addButton_2.setText(_translate("WMML_stk", "Set As Final\n"
"  Estimator"))
        self.label_25.setText(_translate("WMML_stk", "Loaded Model"))
        self.label_26.setText(_translate("WMML_stk", "Meta Model"))
        self.label_20.setText(_translate("WMML_stk", "Final Estimator"))
        self.label.setText(_translate("WMML_stk", "Cross Validation Folds"))
        self.ps_cv.setPlaceholderText(_translate("WMML_stk", "None"))
        self.ps_ps.setText(_translate("WMML_stk", "Enable Passthrough Prediction of Final Estimator"))
        self.groupBox_3.setTitle(_translate("WMML_stk", "Enable Optimazation"))
        self.groupBox_2.setTitle(_translate("WMML_stk", "Parameters"))
        self.label_8.setText(_translate("WMML_stk", "Folds"))
        self.hss_fd.setItemText(0, _translate("WMML_stk", "2"))
        self.hss_fd.setItemText(1, _translate("WMML_stk", "3"))
        self.hss_fd.setItemText(2, _translate("WMML_stk", "5"))
        self.hss_fd.setItemText(3, _translate("WMML_stk", "10"))
        self.hss_fd.setItemText(4, _translate("WMML_stk", "20"))
        self.label_7.setText(_translate("WMML_stk", "Times"))
        self.hss_times.setText(_translate("WMML_stk", "20"))
        self.hss_times.setPlaceholderText(_translate("WMML_stk", "20"))
        self.groupBox_5.setTitle(_translate("WMML_stk", "Algorithms"))
        self.ms_tpeButton.setText(_translate("WMML_stk", "TPE"))
        self.ms_rsaButton.setText(_translate("WMML_stk", "Random"))
        self.ms_saButton.setText(_translate("WMML_stk", "Adaptive"))
        self.groupBox_6.setTitle(_translate("WMML_stk", "Space"))
        self.label_2.setText(_translate("WMML_stk", "Cross Validation Folds"))
        self.label_3.setText(_translate("WMML_stk", "Min"))
        self.label_4.setText(_translate("WMML_stk", "Max"))
        self.hss_cv_min.setText(_translate("WMML_stk", "2"))
        self.hss_cv_min.setPlaceholderText(_translate("WMML_stk", "2"))
        self.hss_cv_max.setText(_translate("WMML_stk", "20"))
        self.hss_cv_max.setPlaceholderText(_translate("WMML_stk", "20"))
        self.hss_ps.setText(_translate("WMML_stk", "Tuning Passthrough\n"
" Prediciton Selection"))
        self.jzsdf.setText(_translate("WMML_stk", "Send Model"))
        self.infobox.setText(_translate("WMML_stk", "....."))
