import sys
import numpy as np
from subwin.wmmlstk import Ui_WMML_stk
from subwin.wmmlsubeng import options
from MLparameters import *
from hyperopt import hp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
import warnings
warnings.filterwarnings('ignore')


class WMML_stk(QWidget, Ui_WMML_stk):
    """
    Stacking Model Define
    Output signal:
    Model object with parameter defined stacking model
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_stk, self).__init__()
        self.setupUi(self)
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.model = Model()
        self.temp_model_bse = None
        self.temp_model_fin = None
        self.bsemd_list = []
        self.bsemd_name_list = []
        self.finmd_list = [('rr', RidgeCV())]
        self.finmd_name_list = ['Ridge Regression']
        self.f_load_f1_2.append("Final Estimator Class:{value}".format(value=self.finmd_list[0][1]))
        self.f_load_f1.append("Final Estimator:{value}".format(value=self.finmd_name_list[0]))
        self.ps_cv.setValidator(QIntValidator())
        self.b_load1.clicked.connect(self.ld_bsemd)
        self.addButton.clicked.connect(self.ad_bsemd)
        self.rmvButton.clicked.connect(self.rm_bsemd)
        self.f_load.clicked.connect(self.ld_finmd)
        self.addButton_2.clicked.connect(self.ad_finmd)
        self.hss_times.setValidator(QIntValidator())
        self.hss_cv_min.setValidator(QIntValidator())
        self.hss_cv_max.setValidator(QIntValidator())

    @staticmethod
    def ad_info_browse(text_widget, info_dict):
        text_widget.clear()
        for key, value in info_dict.items():
            print("Final Estimator:{value}".format(value=value))

    def ld_bsemd(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '',
                                                        'pkcls files(*.pkcls)', options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                self.temp_model_bse = load_model(path_data_position)
                self.b_load_info.clear()
                if self.temp_model_bse.model is not None:
                    self.b_load_info.append('Loaded Model:{}'.format(self.temp_model_bse.model_names))
                    self.b_load_info.append("Model Loaded Complete")
                else:
                    self.b_load_info.append("Model Loaded Failed")
            else:
                self.b_load_info.append("Model Loaded Failed")
        except Exception as e:
            self.b_load_info.append("Model Loaded Failed")
            self.infobox.setText(str(e))

    @staticmethod
    def ad_rm_browse(name_list, md_list, text_browse_1, text_browse_2):
        temp_baemd_list = []
        temp_basmd_class_list = []
        for i in range(len(md_list)):
            temp_baemd_list.append(name_list[i])
            temp_basmd_class_list.append(md_list[i])
        text_browse_1.clear()
        text_browse_2.clear()
        for i in range(len(temp_baemd_list)):
            text_browse_1.append("No.{num} Base Estimator:{value}".format(num=i+1, value=temp_baemd_list[i]))
            text_browse_2.append("No.{num} Base Estimator:{value}".format(num=i+1, value=temp_basmd_class_list[i][1]))

    def ad_bsemd(self):
        temp_bsemd_bse_list = [self.temp_model_bse.model_names_eng, self.temp_model_bse.model]
        self.bsemd_list.append(tuple(temp_bsemd_bse_list))
        self.bsemd_name_list.append(self.temp_model_bse.model_names)
        self.ad_rm_browse(self.bsemd_name_list, self.bsemd_list, self.b_load_b1, self.b_load_b1_2)
        self.b_load_b1.append("Model Added Complete")

    def rm_bsemd(self):
        if len(self.bsemd_list) != 0:
            self.bsemd_list.pop()
            self.ad_rm_browse(self.bsemd_name_list, self.bsemd_list, self.b_load_b1, self.b_load_b1_2)
            self.b_load_b1.append("Model Remove Complete")
        else:
            pass

    def ld_finmd(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'pkcls files(*.pkcls)',
                                                        options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                self.temp_model_fin = load_model(path_data_position)
                self.f_load_info.clear()
                if self.temp_model_fin.model is not None:
                    self.f_load_info.append('Loaded Model：{}'.format(self.temp_model_fin.model_names))
                    self.f_load_info.append("Model Loaded Complete")
                else:
                    self.f_load_info.append("Model Loaded Failed")
            else:
                self.f_load_info.append("Model Loaded Failed")
        except Exception as e:
            self.f_load_info.append("Model Loaded Failed")
            self.infobox.setText(str(e))

    # noinspection PyTypeChecker
    def ad_finmd(self):
        try:
            self.finmd_name_list[0] = self.temp_model_fin.model_names
            temp_list = [self.temp_model_fin.model_names_eng, self.temp_model_fin.model]
            if self.finmd_list:
                self.finmd_list.pop()
            self.finmd_list.append(tuple(temp_list))
            self.f_load_f1_2.clear()
            self.f_load_f1.clear()
            self.f_load_f1_2.append("Final Estimator Class:{value}".format(value=self.finmd_list[0][1]))
            self.f_load_f1.append("Final Estimator:{value}".format(value=self.finmd_name_list[0]))
            self.f_load_info.append("Final Estimator Updated")
        except Exception as e:
            self.f_load_info.append("Updated Failed")
            self.infobox.setText(str(e))

    def get_selection_state(self):
        msb = QMessageBox()
        reply = QMessageBox.question(msb, 'Message', 'Model will be rewrite! Proceed?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.define_model('yes')
        else:
            pass

    @staticmethod
    def redefine_model(model):
        model.model = StackingRegressor(
                estimators=model.parameters['estimators'],
                final_estimator=model.parameters['final_estimator'],
                cv=model.parameters['cv'],
                passthrough=model.parameters['passthrough']
            )
        return model

    def define_model(self, selection):
        if selection == "yes":
            self.model.model_classes = "r"
            self.model.model_names = "Stacking Model"
            self.model.model_names_eng = "stkr"
            try:
                self.model.parameters['cv'] = eval(self.ps_cv.text())
            except Exception as e:
                print(str(e))
                self.ps_cv.setText("None")
                self.model.parameters['cv'] = eval(self.ps_cv.text())
            if self.ps_ps.isChecked():
                self.model.parameters['passthrough'] = True
            else:
                self.model.parameters['passthrough'] = False
            if len(self.bsemd_list) != 0:
                self.model.parameters['estimators'] = self.bsemd_list
            else:
                self.model.parameters['estimators'] = [('rr', RidgeCV())]
            self.model.parameters['final_estimator'] = self.finmd_list[0][1]
            if self.groupBox_3.isChecked():
                self.model.opt_selection = True
                if self.ms_tpeButton.isChecked():
                    self.model.opt_methods = "TPE"
                elif self.ms_rsaButton.isChecked():
                    self.model.opt_methods = "Random"
                else:
                    self.model.opt_methods = "Adaptive"
                try:
                    self.model.opt_times = eval(self.hss_times.text())
                except Exception as e:
                    print(str(e))
                    self.hss_times.setText("20")
                    self.model.opt_times = eval(self.hss_times.text())
                fold_cmbox_text = self.hss_fd.currentText()
                self.model.opt_folds = eval(fold_cmbox_text)
                try:
                    self.model.opt_space['cv'] = hp.randint('cv',
                                                            eval(self.hss_cv_min.text()),
                                                            eval(self.hss_cv_max.text()))
                    self.model.opt_valid_space['cv'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_cv_min.text()),
                                                     stop=eval(self.hss_cv_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_cv_min.setText("2")
                    self.hss_cv_max.setText("20")
                    self.model.opt_space['cv'] = hp.randint('cv',
                                                            eval(self.hss_cv_min.text()),
                                                            eval(self.hss_cv_max.text()))
                    self.model.opt_valid_space['cv'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_cv_min.text()),
                                                     stop=eval(self.hss_cv_max.text()),
                                                     num=10)]
                opt_passthrough_list = []
                if self.hss_ps.isChecked():
                    opt_passthrough_list.append(True)
                    opt_passthrough_list.append(False)
                if len(opt_passthrough_list) != 0:
                    self.model.opt_space['passthrough'] = hp.choice('passthrough', opt_passthrough_list)
                    self.model.opt_compare_space['passthrough'] = opt_passthrough_list
            else:
                self.model.opt_selection = False
            try:
                self.model.model = StackingRegressor(
                    estimators=self.model.parameters['estimators'],
                    final_estimator=self.model.parameters['final_estimator'],
                    cv=self.model.parameters['cv'],
                    passthrough=self.model.parameters['passthrough']
                )
                self.model_signal[object].emit(self.model)
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_stk()
    win.show()
    sys.exit(app.exec_())
