import sys
import numpy as np
from subwin.wmmlrf import Ui_WMML_rf
from MLparameters import *
from hyperopt import hp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class WMML_rf(QWidget, Ui_WMML_rf):
    """
    WMML Random Forest Model Define
    Output signal:
    model: Model object with parameter defined random forest model
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_rf, self).__init__()
        self.setupUi(self)
        self.model = Model()
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.ps_est.setValidator(QIntValidator())
        self.ps_md.setValidator(QIntValidator())
        self.ps_mss.setValidator(QDoubleValidator())
        self.ps_msl.setValidator(QIntValidator())
        self.ps_mwfl.setValidator(QDoubleValidator())
        self.ps_mln.setValidator(QIntValidator())
        self.ps_mid.setValidator(QDoubleValidator())
        self.hss_times.setValidator(QIntValidator())
        self.hss_est_min.setValidator(QIntValidator())
        self.hss_est_max.setValidator(QIntValidator())
        self.hss_md_min.setValidator(QIntValidator())
        self.hss_md_max.setValidator(QIntValidator())
        self.hss_mss_min.setValidator(QIntValidator())
        self.hss_mss_max.setValidator(QIntValidator())
        self.hss_msl_min.setValidator(QIntValidator())
        self.hss_msl_max.setValidator(QIntValidator())
        self.hss_mwfl_min.setValidator(QDoubleValidator())
        self.hss_mwfl_max.setValidator(QDoubleValidator())
        self.hss_mln_min.setValidator(QIntValidator())
        self.hss_mln_max.setValidator(QIntValidator())
        self.hss_mid_min.setValidator(QDoubleValidator())
        self.hss_mid_max.setValidator(QDoubleValidator())

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
        model.model = RandomForestRegressor(
            n_estimators=model.parameters['n_estimators'],
            criterion=model.parameters['criterion'],
            max_depth=model.parameters['max_depth'],
            min_samples_split=model.parameters['min_samples_split'],
            min_samples_leaf=model.parameters['min_samples_leaf'],
            min_weight_fraction_leaf=model.parameters['min_weight_fraction_leaf'],
            max_features=model.parameters['max_features'],
            max_leaf_nodes=model.parameters['max_leaf_nodes'],
            min_impurity_decrease=model.parameters['min_impurity_decrease'],
            bootstrap=model.parameters['bootstrap'])
        return model

    def define_model(self, selection):
        if selection == "yes":
            self.model.model_classes = "r"
            self.model.model_names = "Random Forest"
            self.model.model_names_eng = "rf"
            # Model parameters setting
            try:
                self.model.parameters['n_estimators'] = eval(self.ps_est.text())
            except Exception as e:
                print(str(e))
                self.ps_est.setText("100")
                self.model.parameters['n_estimators'] = eval(self.ps_est.text())
            try:
                self.model.parameters['max_depth'] = eval(self.ps_md.text())
            except Exception as e:
                print(str(e))
                self.ps_md.setText("None")
                self.model.parameters['max_depth'] = eval(self.ps_md.text())
            try:
                self.model.parameters['min_samples_split'] = eval(self.ps_mss.text())
            except Exception as e:
                print(str(e))
                self.ps_mss.setText("2.0")
                self.model.parameters['min_samples_split'] = eval(self.ps_mss.text())
            try:
                self.model.parameters['min_samples_leaf'] = eval(self.ps_msl.text())
            except Exception as e:
                print(str(e))
                self.ps_msl.setText("1")
                self.model.parameters['min_samples_leaf'] = eval(self.ps_msl.text())
            try:
                self.model.parameters['min_weight_fraction_leaf'] = eval(self.ps_mwfl.text())
            except Exception as e:
                print(str(e))
                self.ps_mwfl.setText("1")
                self.model.parameters['min_weight_fraction_leaf'] = eval(self.ps_mwfl.text())
            try:
                self.model.parameters['max_leaf_nodes'] = eval(self.ps_mln.text())
            except Exception as e:
                print(str(e))
                self.ps_mln.setText("None")
                self.model.parameters['max_leaf_nodes'] = eval(self.ps_mln.text())
            try:
                self.model.parameters['min_impurity_decrease'] = eval(self.ps_mid.text())
            except Exception as e:
                print(str(e))
                self.ps_mid.setText("0.0")
                self.model.parameters['min_impurity_decrease'] = eval(self.ps_mid.text())
            if self.c_se.isChecked():
                self.model.parameters['criterion'] = 'mse'
            elif self.c_ase.isChecked():
                self.model.parameters['criterion'] = 'mae'
            else:
                self.model.parameters['criterion'] = 'friedman_mse'
            if self.mf_auto.isChecked():
                self.model.parameters['max_features'] = 'auto'
            elif self.mf_sqrt.isChecked():
                self.model.parameters['max_features'] = 'sqrt'
            else:
                self.model.parameters['max_features'] = 'log2'
            if self.bs_t.isChecked():
                self.model.parameters['bootstrap'] = True
            else:
                self.model.parameters['bootstrap'] = False
            if self.groupBox_2.isChecked():
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
                opt_max_features_list = []
                if self.hss_mf_auto.isChecked():
                    opt_max_features_list.append('auto')
                elif self.hss_mf_log2.isChecked():
                    opt_max_features_list.append('log2')
                elif self.hss_mf_sqrt.isChecked():
                    opt_max_features_list.append('sqrt')
                if len(opt_max_features_list) != 0:
                    self.model.opt_space['max_features'] = hp.choice('max_features', opt_max_features_list)
                    self.model.opt_compare_space['max_features'] = opt_max_features_list
                opt_criterion_list = []
                if self.hss_c_se.isChecked():
                    opt_criterion_list.append('mse')
                elif self.hss_c_ae.isChecked():
                    opt_criterion_list.append('mae')
                elif self.hss_c_poi.isChecked():
                    opt_criterion_list.append('friedman_mse')
                if len(opt_criterion_list) != 0:
                    self.model.opt_space['criterion'] = hp.choice('criterion', opt_criterion_list)
                    self.model.opt_compare_space['criterion'] = opt_criterion_list
                opt_bootstrap_list = []
                if self.hss_bs_tf.isChecked():
                    opt_bootstrap_list.append(True)
                    opt_bootstrap_list.append(False)
                if len(opt_bootstrap_list) != 0:
                    self.model.opt_space['bootstrap'] = hp.choice('bootstrap', opt_bootstrap_list)
                    self.model.opt_compare_space['bootstrap'] = opt_bootstrap_list
                try:
                    self.model.opt_space['n_estimators'] = hp.randint('n_estimators',
                                                                      eval(self.hss_est_min.text()),
                                                                      eval(self.hss_est_max.text()))
                    self.model.opt_valid_space['n_estimators'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_est_min.text()),
                                                     stop=eval(self.hss_est_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_est_min.setText("10")
                    self.hss_est_max.setText("1000")
                    self.model.opt_space['n_estimators'] = hp.randint('n_estimators',
                                                                      eval(self.hss_est_min.text()),
                                                                      eval(self.hss_est_max.text()))
                    self.model.opt_valid_space['n_estimators'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_est_min.text()),
                                                     stop=eval(self.hss_est_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['max_depth'] = hp.randint('max_depth',
                                                                   eval(self.hss_md_min.text()),
                                                                   eval(self.hss_md_max.text()))
                    self.model.opt_valid_space['max_depth'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_md_min.text()),
                                                     stop=eval(self.hss_md_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_md_min.setText("1")
                    self.hss_md_max.setText("32")
                    self.model.opt_space['max_depth'] = hp.randint('max_depth',
                                                                   eval(self.hss_md_min.text()),
                                                                   eval(self.hss_md_max.text()))
                    self.model.opt_valid_space['max_depth'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_md_min.text()),
                                                     stop=eval(self.hss_md_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['min_samples_split'] = hp.randint('min_samples_split',
                                                                           eval(self.hss_mss_min.text()),
                                                                           eval(self.hss_mss_max.text()))
                    self.model.opt_valid_space['min_samples_split'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_mss_min.text()),
                                                     stop=eval(self.hss_mss_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_mss_min.setText("2")
                    self.hss_mss_max.setText("15")
                    self.model.opt_space['min_samples_split'] = hp.randint('min_samples_split',
                                                                           eval(self.hss_mss_min.text()),
                                                                           eval(self.hss_mss_max.text()))
                    self.model.opt_valid_space['min_samples_split'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_mss_min.text()),
                                                     stop=eval(self.hss_mss_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['min_samples_leaf'] = hp.randint('min_samples_leaf',
                                                                          eval(self.hss_msl_min.text()),
                                                                          eval(self.hss_msl_max.text()))
                    self.model.opt_valid_space['min_samples_leaf'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_msl_min.text()),
                                                     stop=eval(self.hss_msl_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_msl_min.setText("2")
                    self.hss_msl_max.setText("15")
                    self.model.opt_space['min_samples_leaf'] = hp.randint('min_samples_leaf',
                                                                          eval(self.hss_msl_min.text()),
                                                                          eval(self.hss_msl_max.text()))
                    self.model.opt_valid_space['min_samples_leaf'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_msl_min.text()),
                                                     stop=eval(self.hss_msl_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['min_weight_fraction_leaf'] = hp.uniform('min_weight_fraction_leaf',
                                                                                  eval(self.hss_mwfl_min.text()),
                                                                                  eval(self.hss_mwfl_max.text()))
                    self.model.opt_valid_space['min_weight_fraction_leaf'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mwfl_min.text()),
                                                       stop=eval(self.hss_mwfl_max.text()),
                                                       num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_mwfl_min.setText("0.0000")
                    self.hss_mwfl_max.setText("0.0001")
                    self.model.opt_space['min_weight_fraction_leaf'] = hp.uniform('min_weight_fraction_leaf',
                                                                                  eval(self.hss_mwfl_min.text()),
                                                                                  eval(self.hss_mwfl_max.text()))
                    self.model.opt_valid_space['min_weight_fraction_leaf'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mwfl_min.text()),
                                                       stop=eval(self.hss_mwfl_max.text()),
                                                       num=10)]
                try:
                    self.model.opt_space['max_leaf_nodes'] = hp.randint('max_leaf_nodes',
                                                                        eval(self.hss_mln_min.text()),
                                                                        eval(self.hss_mln_max.text()))
                    self.model.opt_valid_space['max_leaf_nodes'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_mln_min.text()),
                                                     stop=eval(self.hss_mln_max.text()),
                                                     num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_mln_min.setText("2")
                    self.hss_mln_max.setText("50")
                    self.model.opt_space['max_leaf_nodes'] = hp.randint('max_leaf_nodes',
                                                                        eval(self.hss_mln_min.text()),
                                                                        eval(self.hss_mln_max.text()))
                    self.model.opt_valid_space['max_leaf_nodes'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_mln_min.text()),
                                                     stop=eval(self.hss_mln_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['min_impurity_decrease'] = hp.uniform('min_impurity_decrease',
                                                                               eval(self.hss_mid_min.text()),
                                                                               eval(self.hss_mid_max.text()))
                    self.model.opt_valid_space['min_impurity_decrease'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mid_min.text()),
                                                       stop=eval(self.hss_mid_max.text()),
                                                       num=10)]
                except Exception as e:
                    print(str(e))
                    self.hss_mid_min.setText("0.0000")
                    self.hss_mid_max.setText("0.0001")
                    self.model.opt_space['min_impurity_decrease'] = hp.uniform('min_impurity_decrease',
                                                                               eval(self.hss_mid_min.text()),
                                                                               eval(self.hss_mid_max.text()))
                    self.model.opt_valid_space['min_impurity_decrease'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mid_min.text()),
                                                       stop=eval(self.hss_mid_max.text()),
                                                       num=10)]
            else:
                self.model.opt_selection = False
            try:
                self.model.model = RandomForestRegressor(
                    n_estimators=self.model.parameters['n_estimators'],
                    criterion=self.model.parameters['criterion'],
                    max_depth=self.model.parameters['max_depth'],
                    min_samples_split=self.model.parameters['min_samples_split'],
                    min_samples_leaf=self.model.parameters['min_samples_leaf'],
                    min_weight_fraction_leaf=self.model.parameters['min_weight_fraction_leaf'],
                    max_features=self.model.parameters['max_features'],
                    max_leaf_nodes=self.model.parameters['max_leaf_nodes'],
                    min_impurity_decrease=self.model.parameters['min_impurity_decrease'],
                    bootstrap=self.model.parameters['bootstrap'])
                self.model_signal[object].emit(self.model)
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_rf()
    win.show()
    sys.exit(app.exec_())
