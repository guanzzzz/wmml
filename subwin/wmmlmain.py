from PyQt5 import QtCore, QtGui, QtWidgets
#import ctypes
#ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")


# noinspection PyUnresolvedReferences
class Ui_WMML_mainwindow(object):
    def setupUi(self, WMML_mainwindow):
        WMML_mainwindow.setObjectName("WMML_mainwindow")
        WMML_mainwindow.resize(1440, 960)
        font = QtGui.QFont()
        font.setFamily("Arial")
        WMML_mainwindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon/DMS.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        WMML_mainwindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(WMML_mainwindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.subw_ma = QtWidgets.QMdiArea(self.centralwidget)
        self.subw_ma.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.subw_ma.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        brush = QtGui.QBrush(QtGui.QColor(203, 202, 204))
        brush.setStyle(QtCore.Qt.SolidPattern)
        self.subw_ma.setBackground(brush)
        self.subw_ma.setActivationOrder(QtWidgets.QMdiArea.CreationOrder)
        self.subw_ma.setViewMode(QtWidgets.QMdiArea.SubWindowView)
        self.subw_ma.setTabsMovable(False)
        self.subw_ma.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.subw_ma.setObjectName("subw_ma")
        self.gridLayout.addWidget(self.subw_ma, 0, 0, 1, 1)
        WMML_mainwindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(WMML_mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1440, 23))
        self.menubar.setObjectName("menubar")
        WMML_mainwindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(WMML_mainwindow)
        self.statusbar.setObjectName("statusbar")
        WMML_mainwindow.setStatusBar(self.statusbar)
        self.tool_dw = QtWidgets.QDockWidget(WMML_mainwindow)
        self.tool_dw.setAutoFillBackground(False)
        self.tool_dw.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.tool_dw.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.tool_dw.setObjectName("tool_dw")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolBox = QtWidgets.QToolBox(self.dockWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QtCore.QSize(160, 251))
        self.toolBox.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.toolBox.setFont(font)
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 160, 312))
        self.page.setObjectName("page")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.page)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.fe_corr_btn = QtWidgets.QPushButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fe_corr_btn.sizePolicy().hasHeightForWidth())
        self.fe_corr_btn.setSizePolicy(sizePolicy)
        self.fe_corr_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.fe_corr_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.fe_corr_btn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon/corre.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.fe_corr_btn.setIcon(icon1)
        self.fe_corr_btn.setIconSize(QtCore.QSize(60, 90))
        self.fe_corr_btn.setFlat(True)
        self.fe_corr_btn.setObjectName("fe_corr_btn")
        self.gridLayout_3.addWidget(self.fe_corr_btn, 1, 0, 1, 1)
        self.jz_ld_btn = QtWidgets.QPushButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.jz_ld_btn.sizePolicy().hasHeightForWidth())
        self.jz_ld_btn.setSizePolicy(sizePolicy)
        self.jz_ld_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.jz_ld_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.jz_ld_btn.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon/csvinput.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.jz_ld_btn.setIcon(icon2)
        self.jz_ld_btn.setIconSize(QtCore.QSize(60, 90))
        self.jz_ld_btn.setFlat(True)
        self.jz_ld_btn.setObjectName("jz_ld_btn")
        self.gridLayout_3.addWidget(self.jz_ld_btn, 0, 0, 1, 1)
        self.ml_r_lr_btn = QtWidgets.QPushButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_lr_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_lr_btn.setSizePolicy(sizePolicy)
        self.ml_r_lr_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_lr_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_lr_btn.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon/SR.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_lr_btn.setIcon(icon3)
        self.ml_r_lr_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_lr_btn.setFlat(True)
        self.ml_r_lr_btn.setObjectName("ml_r_lr_btn")
        self.gridLayout_3.addWidget(self.ml_r_lr_btn, 2, 0, 1, 1)
        self.ml_u_pca_btn = QtWidgets.QPushButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_u_pca_btn.sizePolicy().hasHeightForWidth())
        self.ml_u_pca_btn.setSizePolicy(sizePolicy)
        self.ml_u_pca_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.ml_u_pca_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_u_pca_btn.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icon/pca.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_u_pca_btn.setIcon(icon4)
        self.ml_u_pca_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_u_pca_btn.setFlat(True)
        self.ml_u_pca_btn.setObjectName("ml_u_pca_btn")
        self.gridLayout_3.addWidget(self.ml_u_pca_btn, 1, 1, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.fe_prep_btn = QtWidgets.QPushButton(self.page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fe_prep_btn.sizePolicy().hasHeightForWidth())
        self.fe_prep_btn.setSizePolicy(sizePolicy)
        self.fe_prep_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.fe_prep_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.fe_prep_btn.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("icon/processdata.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.fe_prep_btn.setIcon(icon5)
        self.fe_prep_btn.setIconSize(QtCore.QSize(70, 90))
        self.fe_prep_btn.setFlat(True)
        self.fe_prep_btn.setObjectName("fe_prep_btn")
        self.gridLayout_3.addWidget(self.fe_prep_btn, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("icon/input_data.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox.addItem(self.page, icon6, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.toolBox_2 = QtWidgets.QToolBox(self.dockWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.toolBox_2.setFont(font)
        self.toolBox_2.setObjectName("toolBox_2")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 160, 312))
        self.page_2.setObjectName("page_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.ml_r_knn_btn = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_knn_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_knn_btn.setSizePolicy(sizePolicy)
        self.ml_r_knn_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_knn_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_knn_btn.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("icon/KNN.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_knn_btn.setIcon(icon7)
        self.ml_r_knn_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_knn_btn.setFlat(True)
        self.ml_r_knn_btn.setObjectName("ml_r_knn_btn")
        self.gridLayout_6.addWidget(self.ml_r_knn_btn, 1, 0, 1, 1)
        self.ml_r_mlp_btn = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_mlp_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_mlp_btn.setSizePolicy(sizePolicy)
        self.ml_r_mlp_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_mlp_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_mlp_btn.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("icon/NN.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_mlp_btn.setIcon(icon8)
        self.ml_r_mlp_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_mlp_btn.setFlat(True)
        self.ml_r_mlp_btn.setObjectName("ml_r_mlp_btn")
        self.gridLayout_6.addWidget(self.ml_r_mlp_btn, 1, 1, 1, 1)
        self.ml_r_rf_btn = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_rf_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_rf_btn.setSizePolicy(sizePolicy)
        self.ml_r_rf_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_rf_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_rf_btn.setText("")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("icon/RF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_rf_btn.setIcon(icon9)
        self.ml_r_rf_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_rf_btn.setFlat(True)
        self.ml_r_rf_btn.setObjectName("ml_r_rf_btn")
        self.gridLayout_6.addWidget(self.ml_r_rf_btn, 0, 0, 1, 1)
        self.ml_r_gbdt_btn = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_gbdt_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_gbdt_btn.setSizePolicy(sizePolicy)
        self.ml_r_gbdt_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_gbdt_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_gbdt_btn.setText("")
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("icon/GBDT.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_gbdt_btn.setIcon(icon10)
        self.ml_r_gbdt_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_gbdt_btn.setFlat(True)
        self.ml_r_gbdt_btn.setObjectName("ml_r_gbdt_btn")
        self.gridLayout_6.addWidget(self.ml_r_gbdt_btn, 0, 1, 1, 1)
        self.ml_r_stk_btn = QtWidgets.QPushButton(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_r_stk_btn.sizePolicy().hasHeightForWidth())
        self.ml_r_stk_btn.setSizePolicy(sizePolicy)
        self.ml_r_stk_btn.setMinimumSize(QtCore.QSize(60, 88))
        self.ml_r_stk_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_r_stk_btn.setText("")
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("icon/STK.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_r_stk_btn.setIcon(icon11)
        self.ml_r_stk_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_r_stk_btn.setFlat(True)
        self.ml_r_stk_btn.setObjectName("ml_r_stk_btn")
        self.gridLayout_6.addWidget(self.ml_r_stk_btn, 2, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("icon/MLTE.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox_2.addItem(self.page_2, icon12, "")
        self.verticalLayout.addWidget(self.toolBox_2)
        self.toolBox_3 = QtWidgets.QToolBox(self.dockWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.toolBox_3.setFont(font)
        self.toolBox_3.setObjectName("toolBox_3")
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setGeometry(QtCore.QRect(0, 0, 160, 114))
        self.page_6.setObjectName("page_6")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.page_6)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ml_el_btn = QtWidgets.QPushButton(self.page_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_el_btn.sizePolicy().hasHeightForWidth())
        self.ml_el_btn.setSizePolicy(sizePolicy)
        self.ml_el_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.ml_el_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_el_btn.setText("")
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap("icon/ts.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_el_btn.setIcon(icon13)
        self.ml_el_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_el_btn.setFlat(True)
        self.ml_el_btn.setObjectName("ml_el_btn")
        self.horizontalLayout.addWidget(self.ml_el_btn)
        self.ml_op_bg_btn = QtWidgets.QPushButton(self.page_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ml_op_bg_btn.sizePolicy().hasHeightForWidth())
        self.ml_op_bg_btn.setSizePolicy(sizePolicy)
        self.ml_op_bg_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.ml_op_bg_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.ml_op_bg_btn.setText("")
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap("icon/opt.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ml_op_bg_btn.setIcon(icon14)
        self.ml_op_bg_btn.setIconSize(QtCore.QSize(60, 90))
        self.ml_op_bg_btn.setFlat(True)
        self.ml_op_bg_btn.setObjectName("ml_op_bg_btn")
        self.horizontalLayout.addWidget(self.ml_op_bg_btn)
        self.gridLayout_13.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        icon15 = QtGui.QIcon()
        icon15.addPixmap(QtGui.QPixmap("icon/sgdc.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox_3.addItem(self.page_6, icon15, "")
        self.page_7 = QtWidgets.QWidget()
        self.page_7.setGeometry(QtCore.QRect(0, 0, 160, 114))
        self.page_7.setObjectName("page_7")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.page_7)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.gridLayout_16 = QtWidgets.QGridLayout()
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.tr_pred_btn = QtWidgets.QPushButton(self.page_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tr_pred_btn.sizePolicy().hasHeightForWidth())
        self.tr_pred_btn.setSizePolicy(sizePolicy)
        self.tr_pred_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.tr_pred_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.tr_pred_btn.setText("")
        icon16 = QtGui.QIcon()
        icon16.addPixmap(QtGui.QPixmap("icon/pred.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tr_pred_btn.setIcon(icon16)
        self.tr_pred_btn.setIconSize(QtCore.QSize(60, 90))
        self.tr_pred_btn.setFlat(True)
        self.tr_pred_btn.setObjectName("tr_pred_btn")
        self.gridLayout_16.addWidget(self.tr_pred_btn, 0, 0, 1, 1)
        self.tr_ldmd_btn = QtWidgets.QPushButton(self.page_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tr_ldmd_btn.sizePolicy().hasHeightForWidth())
        self.tr_ldmd_btn.setSizePolicy(sizePolicy)
        self.tr_ldmd_btn.setMinimumSize(QtCore.QSize(60, 90))
        self.tr_ldmd_btn.setMaximumSize(QtCore.QSize(60, 90))
        self.tr_ldmd_btn.setText("")
        icon17 = QtGui.QIcon()
        icon17.addPixmap(QtGui.QPixmap("icon/loadmodel.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tr_ldmd_btn.setIcon(icon17)
        self.tr_ldmd_btn.setIconSize(QtCore.QSize(60, 90))
        self.tr_ldmd_btn.setFlat(True)
        self.tr_ldmd_btn.setObjectName("tr_ldmd_btn")
        self.gridLayout_16.addWidget(self.tr_ldmd_btn, 0, 1, 1, 1)
        self.gridLayout_15.addLayout(self.gridLayout_16, 0, 0, 1, 1)
        icon18 = QtGui.QIcon()
        icon18.addPixmap(QtGui.QPixmap("icon/1sft.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox_3.addItem(self.page_7, icon18, "")
        self.verticalLayout.addWidget(self.toolBox_3)
        self.verticalLayout.setStretch(0, 2)
        self.verticalLayout.setStretch(1, 2)
        self.verticalLayout.setStretch(2, 1)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.tool_dw.setWidget(self.dockWidgetContents)
        WMML_mainwindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.tool_dw)

        self.retranslateUi(WMML_mainwindow)
        self.toolBox.setCurrentIndex(0)
        self.toolBox.layout().setSpacing(0)
        self.toolBox_2.setCurrentIndex(0)
        self.toolBox_2.layout().setSpacing(0)
        self.toolBox_3.setCurrentIndex(0)
        self.toolBox_3.layout().setSpacing(0)
        QtCore.QMetaObject.connectSlotsByName(WMML_mainwindow)

    def retranslateUi(self, WMML_mainwindow):
        _translate = QtCore.QCoreApplication.translate
        WMML_mainwindow.setWindowTitle(_translate("WMML_mainwindow", "WMML"))
        self.tool_dw.setWindowTitle(_translate("WMML_mainwindow", "WMML Tool Box"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("WMML_mainwindow", "Data and Feature"))
        self.toolBox_2.setItemText(self.toolBox_2.indexOf(self.page_2), _translate("WMML_mainwindow",
                                                                                   "ML Algorithms"))
        self.toolBox_3.setItemText(self.toolBox_3.indexOf(self.page_6), _translate("WMML_mainwindow",
                                                                                   "ML Train and Test"))
        self.toolBox_3.setItemText(self.toolBox_3.indexOf(self.page_7), _translate("WMML_mainwindow",
                                                                                   "ML Prediction"))
