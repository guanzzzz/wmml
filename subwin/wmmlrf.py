from PyQt5 import QtCore, QtGui, QtWidgets


# noinspection PyUnresolvedReferences
class Ui_WMML_rf(object):
    def setupUi(self, WMML_rf):
        WMML_rf.setObjectName("WMML_rf")
        WMML_rf.resize(835, 523)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WMML_rf.sizePolicy().hasHeightForWidth())
        WMML_rf.setSizePolicy(sizePolicy)
        WMML_rf.setMinimumSize(QtCore.QSize(835, 523))
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(WMML_rf)
        self.horizontalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.groupBox = QtWidgets.QGroupBox(WMML_rf)
        self.groupBox.setMinimumSize(QtCore.QSize(269, 423))
        self.groupBox.setMaximumSize(QtCore.QSize(269, 423))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.ps_est = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_est.setFont(font)
        self.ps_est.setObjectName("ps_est")
        self.gridLayout.addWidget(self.ps_est, 0, 1, 1, 1)
        self.ps_mwfl = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_mwfl.setFont(font)
        self.ps_mwfl.setObjectName("ps_mwfl")
        self.gridLayout.addWidget(self.ps_mwfl, 4, 1, 1, 1)
        self.ps_mln = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_mln.setFont(font)
        self.ps_mln.setText("")
        self.ps_mln.setObjectName("ps_mln")
        self.gridLayout.addWidget(self.ps_mln, 5, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.ps_md = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_md.setFont(font)
        self.ps_md.setText("")
        self.ps_md.setObjectName("ps_md")
        self.gridLayout.addWidget(self.ps_md, 1, 1, 1, 1)
        self.ps_mss = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_mss.setFont(font)
        self.ps_mss.setObjectName("ps_mss")
        self.gridLayout.addWidget(self.ps_mss, 2, 1, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 5, 0, 1, 1)
        self.ps_msl = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_msl.setFont(font)
        self.ps_msl.setObjectName("ps_msl")
        self.gridLayout.addWidget(self.ps_msl, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 6, 0, 1, 1)
        self.ps_mid = QtWidgets.QLineEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ps_mid.setFont(font)
        self.ps_mid.setObjectName("ps_mid")
        self.gridLayout.addWidget(self.ps_mid, 6, 1, 1, 1)
        self.verticalLayout_6.addLayout(self.gridLayout)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_9.setFont(font)
        self.groupBox_9.setObjectName("groupBox_9")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_9)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.bs_f = QtWidgets.QRadioButton(self.groupBox_9)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.bs_f.setFont(font)
        self.bs_f.setObjectName("bs_f")
        self.verticalLayout_9.addWidget(self.bs_f)
        self.bs_t = QtWidgets.QRadioButton(self.groupBox_9)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.bs_t.setFont(font)
        self.bs_t.setChecked(True)
        self.bs_t.setObjectName("bs_t")
        self.verticalLayout_9.addWidget(self.bs_t)
        self.horizontalLayout_5.addWidget(self.groupBox_9)
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_10.setFont(font)
        self.groupBox_10.setObjectName("groupBox_10")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_10)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.c_se = QtWidgets.QRadioButton(self.groupBox_10)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.c_se.setFont(font)
        self.c_se.setChecked(True)
        self.c_se.setObjectName("c_se")
        self.horizontalLayout_2.addWidget(self.c_se)
        self.c_ase = QtWidgets.QRadioButton(self.groupBox_10)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.c_ase.setFont(font)
        self.c_ase.setObjectName("c_ase")
        self.horizontalLayout_2.addWidget(self.c_ase)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.c_p = QtWidgets.QRadioButton(self.groupBox_10)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.c_p.setFont(font)
        self.c_p.setObjectName("c_p")
        self.verticalLayout_5.addWidget(self.c_p)
        self.horizontalLayout_5.addWidget(self.groupBox_10)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.mf_log = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.mf_log.setFont(font)
        self.mf_log.setObjectName("mf_log")
        self.horizontalLayout_4.addWidget(self.mf_log)
        self.mf_sqrt = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.mf_sqrt.setFont(font)
        self.mf_sqrt.setObjectName("mf_sqrt")
        self.horizontalLayout_4.addWidget(self.mf_sqrt)
        self.mf_auto = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.mf_auto.setFont(font)
        self.mf_auto.setChecked(True)
        self.mf_auto.setObjectName("mf_auto")
        self.horizontalLayout_4.addWidget(self.mf_auto)
        self.verticalLayout_6.addWidget(self.groupBox_8)
        self.verticalLayout_8.addWidget(self.groupBox)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_8.addItem(spacerItem)
        self.jzsdf = QtWidgets.QPushButton(WMML_rf)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.jzsdf.setFont(font)
        self.jzsdf.setObjectName("jzsdf")
        self.verticalLayout_8.addWidget(self.jzsdf)
        self.infobox = QtWidgets.QLabel(WMML_rf)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.infobox.setFont(font)
        self.infobox.setObjectName("infobox")
        self.verticalLayout_8.addWidget(self.infobox)
        self.horizontalLayout_7.addLayout(self.verticalLayout_8)
        self.groupBox_2 = QtWidgets.QGroupBox(WMML_rf)
        self.groupBox_2.setMinimumSize(QtCore.QSize(531, 501))
        self.groupBox_2.setMaximumSize(QtCore.QSize(531, 501))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setCheckable(True)
        self.groupBox_2.setChecked(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.groupBox_11 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_11.setFont(font)
        self.groupBox_11.setObjectName("groupBox_11")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_11)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.hss_c_se = QtWidgets.QCheckBox(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_c_se.setFont(font)
        self.hss_c_se.setChecked(True)
        self.hss_c_se.setObjectName("hss_c_se")
        self.verticalLayout.addWidget(self.hss_c_se)
        self.hss_c_ae = QtWidgets.QCheckBox(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_c_ae.setFont(font)
        self.hss_c_ae.setChecked(True)
        self.hss_c_ae.setObjectName("hss_c_ae")
        self.verticalLayout.addWidget(self.hss_c_ae)
        self.hss_c_poi = QtWidgets.QCheckBox(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_c_poi.setFont(font)
        self.hss_c_poi.setChecked(True)
        self.hss_c_poi.setObjectName("hss_c_poi")
        self.verticalLayout.addWidget(self.hss_c_poi)
        self.verticalLayout_4.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.hss_mf_auto = QtWidgets.QCheckBox(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mf_auto.setFont(font)
        self.hss_mf_auto.setChecked(True)
        self.hss_mf_auto.setObjectName("hss_mf_auto")
        self.verticalLayout_2.addWidget(self.hss_mf_auto)
        self.hss_mf_log2 = QtWidgets.QCheckBox(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mf_log2.setFont(font)
        self.hss_mf_log2.setChecked(True)
        self.hss_mf_log2.setObjectName("hss_mf_log2")
        self.verticalLayout_2.addWidget(self.hss_mf_log2)
        self.hss_mf_sqrt = QtWidgets.QCheckBox(self.groupBox_5)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mf_sqrt.setFont(font)
        self.hss_mf_sqrt.setChecked(True)
        self.hss_mf_sqrt.setObjectName("hss_mf_sqrt")
        self.verticalLayout_2.addWidget(self.hss_mf_sqrt)
        self.verticalLayout_4.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.hss_bs_tf = QtWidgets.QCheckBox(self.groupBox_6)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_bs_tf.setFont(font)
        self.hss_bs_tf.setChecked(True)
        self.hss_bs_tf.setObjectName("hss_bs_tf")
        self.verticalLayout_3.addWidget(self.hss_bs_tf)
        self.verticalLayout_4.addWidget(self.groupBox_6)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_13 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 5, 0, 1, 1)
        self.hss_mln_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mln_min.setFont(font)
        self.hss_mln_min.setObjectName("hss_mln_min")
        self.gridLayout_2.addWidget(self.hss_mln_min, 6, 1, 1, 1)
        self.hss_est_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_est_max.setFont(font)
        self.hss_est_max.setObjectName("hss_est_max")
        self.gridLayout_2.addWidget(self.hss_est_max, 1, 2, 1, 1)
        self.hss_mss_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mss_max.setFont(font)
        self.hss_mss_max.setObjectName("hss_mss_max")
        self.gridLayout_2.addWidget(self.hss_mss_max, 3, 2, 1, 1)
        self.hss_mss_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mss_min.setFont(font)
        self.hss_mss_min.setObjectName("hss_mss_min")
        self.gridLayout_2.addWidget(self.hss_mss_min, 3, 1, 1, 1)
        self.hss_mln_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mln_max.setFont(font)
        self.hss_mln_max.setObjectName("hss_mln_max")
        self.gridLayout_2.addWidget(self.hss_mln_max, 6, 2, 1, 1)
        self.hss_mid_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mid_min.setFont(font)
        self.hss_mid_min.setObjectName("hss_mid_min")
        self.gridLayout_2.addWidget(self.hss_mid_min, 7, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 3, 0, 1, 1)
        self.hss_mwfl_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mwfl_max.setFont(font)
        self.hss_mwfl_max.setObjectName("hss_mwfl_max")
        self.gridLayout_2.addWidget(self.hss_mwfl_max, 5, 2, 1, 1)
        self.hss_md_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_md_max.setFont(font)
        self.hss_md_max.setObjectName("hss_md_max")
        self.gridLayout_2.addWidget(self.hss_md_max, 2, 2, 1, 1)
        self.hss_md_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_md_min.setFont(font)
        self.hss_md_min.setObjectName("hss_md_min")
        self.gridLayout_2.addWidget(self.hss_md_min, 2, 1, 1, 1)
        self.hss_msl_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_msl_max.setFont(font)
        self.hss_msl_max.setObjectName("hss_msl_max")
        self.gridLayout_2.addWidget(self.hss_msl_max, 4, 2, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 6, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 0, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 7, 0, 1, 1)
        self.hss_mid_max = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mid_max.setFont(font)
        self.hss_mid_max.setObjectName("hss_mid_max")
        self.gridLayout_2.addWidget(self.hss_mid_max, 7, 2, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 1, 0, 1, 1)
        self.hss_msl_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_msl_min.setFont(font)
        self.hss_msl_min.setObjectName("hss_msl_min")
        self.gridLayout_2.addWidget(self.hss_msl_min, 4, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 2, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 4, 0, 1, 1)
        self.hss_mwfl_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_mwfl_min.setFont(font)
        self.hss_mwfl_min.setObjectName("hss_mwfl_min")
        self.gridLayout_2.addWidget(self.hss_mwfl_min, 5, 1, 1, 1)
        self.hss_est_min = QtWidgets.QLineEdit(self.groupBox_11)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_est_min.setFont(font)
        self.hss_est_min.setObjectName("hss_est_min")
        self.gridLayout_2.addWidget(self.hss_est_min, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_2)
        self.verticalLayout_7.addWidget(self.groupBox_11)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.hss_fd = QtWidgets.QComboBox(self.groupBox_3)
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
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.hss_times = QtWidgets.QLineEdit(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.hss_times.setFont(font)
        self.hss_times.setObjectName("hss_times")
        self.gridLayout_4.addWidget(self.hss_times, 0, 1, 1, 1)
        self.gridLayout_4.setColumnMinimumWidth(0, 1)
        self.gridLayout_4.setColumnMinimumWidth(1, 1)
        self.gridLayout_4.setRowMinimumHeight(0, 1)
        self.gridLayout_4.setRowMinimumHeight(1, 1)
        self.horizontalLayout_3.addLayout(self.gridLayout_4)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.groupBox_7.setFont(font)
        self.groupBox_7.setCheckable(False)
        self.groupBox_7.setChecked(False)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.ms_tpeButton = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_tpeButton.setFont(font)
        self.ms_tpeButton.setChecked(True)
        self.ms_tpeButton.setObjectName("ms_tpeButton")
        self.horizontalLayout_6.addWidget(self.ms_tpeButton)
        self.ms_saButton = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_saButton.setFont(font)
        self.ms_saButton.setObjectName("ms_saButton")
        self.horizontalLayout_6.addWidget(self.ms_saButton)
        self.ms_rsaButton = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.ms_rsaButton.setFont(font)
        self.ms_rsaButton.setObjectName("ms_rsaButton")
        self.horizontalLayout_6.addWidget(self.ms_rsaButton)
        self.horizontalLayout_3.addWidget(self.groupBox_7)
        self.verticalLayout_7.addWidget(self.groupBox_3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem1)
        self.horizontalLayout_7.addWidget(self.groupBox_2)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)

        self.retranslateUi(WMML_rf)
        QtCore.QMetaObject.connectSlotsByName(WMML_rf)

    def retranslateUi(self, WMML_rf):
        _translate = QtCore.QCoreApplication.translate
        WMML_rf.setWindowTitle(_translate("WMML_rf", "WMMLrf"))
        self.groupBox.setTitle(_translate("WMML_rf", "Model Parameters"))
        self.label_6.setText(_translate("WMML_rf", "Min Leaf Weight Fraction"))
        self.label_4.setText(_translate("WMML_rf", "Min Samples Split"))
        self.label_2.setText(_translate("WMML_rf", "Number of Tree"))
        self.ps_est.setText(_translate("WMML_rf", "100"))
        self.ps_est.setPlaceholderText(_translate("WMML_rf", "100"))
        self.ps_mwfl.setText(_translate("WMML_rf", "0.0000"))
        self.ps_mwfl.setPlaceholderText(_translate("WMML_rf", "0.0000"))
        self.ps_mln.setPlaceholderText(_translate("WMML_rf", "None"))
        self.label_3.setText(_translate("WMML_rf", "Max Depth"))
        self.ps_md.setPlaceholderText(_translate("WMML_rf", "None"))
        self.ps_mss.setText(_translate("WMML_rf", "2"))
        self.ps_mss.setPlaceholderText(_translate("WMML_rf", "2"))
        self.label_22.setText(_translate("WMML_rf", "Max Leaf Nodes"))
        self.ps_msl.setText(_translate("WMML_rf", "1"))
        self.ps_msl.setPlaceholderText(_translate("WMML_rf", "1"))
        self.label_5.setText(_translate("WMML_rf", "Min Samples Leaf"))
        self.label_21.setText(_translate("WMML_rf", "Min Impurity Decrease"))
        self.ps_mid.setText(_translate("WMML_rf", "0.0"))
        self.ps_mid.setPlaceholderText(_translate("WMML_rf", "0.0"))
        self.groupBox_9.setTitle(_translate("WMML_rf", "Bootstrap"))
        self.bs_f.setText(_translate("WMML_rf", "No"))
        self.bs_t.setText(_translate("WMML_rf", "Yes"))
        self.groupBox_10.setTitle(_translate("WMML_rf", "Criterion"))
        self.c_se.setText(_translate("WMML_rf", "MSE"))
        self.c_ase.setText(_translate("WMML_rf", "MAE"))
        self.c_p.setText(_translate("WMML_rf", "Friedman MSE"))
        self.groupBox_8.setTitle(_translate("WMML_rf", "Max Features"))
        self.mf_log.setText(_translate("WMML_rf", "Log2"))
        self.mf_sqrt.setText(_translate("WMML_rf", "Sqrt"))
        self.mf_auto.setText(_translate("WMML_rf", "Auto"))
        self.jzsdf.setText(_translate("WMML_rf", "Send Model"))
        self.infobox.setText(_translate("WMML_rf", "....."))
        self.groupBox_2.setTitle(_translate("WMML_rf", "Enable Optimazation"))
        self.groupBox_11.setTitle(_translate("WMML_rf", "Space"))
        self.groupBox_4.setTitle(_translate("WMML_rf", "Criterion"))
        self.hss_c_se.setText(_translate("WMML_rf", "MSE"))
        self.hss_c_ae.setText(_translate("WMML_rf", "MAE"))
        self.hss_c_poi.setText(_translate("WMML_rf", "Friedman MSE"))
        self.groupBox_5.setTitle(_translate("WMML_rf", "Max Features"))
        self.hss_mf_auto.setText(_translate("WMML_rf", "Auto"))
        self.hss_mf_log2.setText(_translate("WMML_rf", "Log2"))
        self.hss_mf_sqrt.setText(_translate("WMML_rf", "Sqrt"))
        self.groupBox_6.setTitle(_translate("WMML_rf", "Bootstrap"))
        self.hss_bs_tf.setText(_translate("WMML_rf", "Yes/No"))
        self.label_13.setText(_translate("WMML_rf", "Min Leaf Weight Fraction"))
        self.hss_mln_min.setText(_translate("WMML_rf", "2"))
        self.hss_mln_min.setPlaceholderText(_translate("WMML_rf", "2"))
        self.hss_est_max.setText(_translate("WMML_rf", "1000"))
        self.hss_est_max.setPlaceholderText(_translate("WMML_rf", "1000"))
        self.hss_mss_max.setText(_translate("WMML_rf", "15"))
        self.hss_mss_max.setPlaceholderText(_translate("WMML_rf", "15"))
        self.hss_mss_min.setText(_translate("WMML_rf", "2"))
        self.hss_mss_min.setPlaceholderText(_translate("WMML_rf", "2"))
        self.hss_mln_max.setText(_translate("WMML_rf", "50"))
        self.hss_mln_max.setPlaceholderText(_translate("WMML_rf", "50"))
        self.hss_mid_min.setText(_translate("WMML_rf", "0.0000"))
        self.hss_mid_min.setPlaceholderText(_translate("WMML_rf", "0.0000"))
        self.label_11.setText(_translate("WMML_rf", "Min Samples Split"))
        self.hss_mwfl_max.setText(_translate("WMML_rf", "0.0001"))
        self.hss_mwfl_max.setPlaceholderText(_translate("WMML_rf", "0.0001"))
        self.hss_md_max.setText(_translate("WMML_rf", "32"))
        self.hss_md_max.setPlaceholderText(_translate("WMML_rf", "32"))
        self.hss_md_min.setText(_translate("WMML_rf", "1"))
        self.hss_md_min.setPlaceholderText(_translate("WMML_rf", "1"))
        self.hss_msl_max.setText(_translate("WMML_rf", "15"))
        self.hss_msl_max.setPlaceholderText(_translate("WMML_rf", "15"))
        self.label_19.setText(_translate("WMML_rf", "Max Leaf Nodes"))
        self.label_16.setText(_translate("WMML_rf", "Min"))
        self.label_20.setText(_translate("WMML_rf", "Min Impurity Decrease"))
        self.hss_mid_max.setText(_translate("WMML_rf", "0.0001"))
        self.hss_mid_max.setPlaceholderText(_translate("WMML_rf", "0.0001"))
        self.label_17.setText(_translate("WMML_rf", "Max"))
        self.label_9.setText(_translate("WMML_rf", "Number of Tree"))
        self.hss_msl_min.setText(_translate("WMML_rf", "2"))
        self.hss_msl_min.setPlaceholderText(_translate("WMML_rf", "2"))
        self.label_10.setText(_translate("WMML_rf", "Max Depth"))
        self.label_12.setText(_translate("WMML_rf", "Min Samples Leaf"))
        self.hss_mwfl_min.setText(_translate("WMML_rf", "0.0000"))
        self.hss_mwfl_min.setPlaceholderText(_translate("WMML_rf", "0.0000"))
        self.hss_est_min.setText(_translate("WMML_rf", "10"))
        self.hss_est_min.setPlaceholderText(_translate("WMML_rf", "10"))
        self.groupBox_3.setTitle(_translate("WMML_rf", "Parameters"))
        self.label_8.setText(_translate("WMML_rf", "Folds"))
        self.hss_fd.setItemText(0, _translate("WMML_rf", "2"))
        self.hss_fd.setItemText(1, _translate("WMML_rf", "3"))
        self.hss_fd.setItemText(2, _translate("WMML_rf", "5"))
        self.hss_fd.setItemText(3, _translate("WMML_rf", "10"))
        self.hss_fd.setItemText(4, _translate("WMML_rf", "20"))
        self.label_7.setText(_translate("WMML_rf", "Times"))
        self.hss_times.setText(_translate("WMML_rf", "20"))
        self.hss_times.setPlaceholderText(_translate("WMML_rf", "20"))
        self.groupBox_7.setTitle(_translate("WMML_rf", "Algorithms"))
        self.ms_tpeButton.setText(_translate("WMML_rf", "TPE"))
        self.ms_saButton.setText(_translate("WMML_rf", "Adaptive"))
        self.ms_rsaButton.setText(_translate("WMML_rf", "Random"))