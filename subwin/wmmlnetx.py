import sys
import numpy as np
from subwin.wmmlnet import Ui_WMML_nn
from MLparameters import *
from hyperopt import hp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


class WMML_nn(QWidget, Ui_WMML_nn):
    """
    WMML Neutral Network Model Define
    Output signal:
    model: Model object with parameter defined neutral network model
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_nn, self).__init__()
        self.setupUi(self)
        self.model = Model()
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.ps_hls.setValidator(QIntValidator())
        self.ps_hls_2.setValidator(QIntValidator())
        self.ps_hls_3.setValidator(QIntValidator())
        self.ps_hls_4.setValidator(QIntValidator())
        self.ps_hls_5.setValidator(QIntValidator())
        self.ps_bs_m.setValidator(QIntValidator())
        self.ps_alpha.setValidator(QDoubleValidator())
        self.ps_tol.setValidator(QDoubleValidator())
        self.ps_iter.setValidator(QIntValidator())
        self.hss_times.setValidator(QIntValidator())
        self.hss_hls_min.setValidator(QIntValidator())
        self.hss_hls_min_2.setValidator(QIntValidator())
        self.hss_hls_min_3.setValidator(QIntValidator())
        self.hss_hls_min_4.setValidator(QIntValidator())
        self.hss_hls_min_5.setValidator(QIntValidator())
        self.hss_hls_max.setValidator(QIntValidator())
        self.hss_hls_max_2.setValidator(QIntValidator())
        self.hss_hls_max_3.setValidator(QIntValidator())
        self.hss_hls_max_4.setValidator(QIntValidator())
        self.hss_hls_max_5.setValidator(QIntValidator())
        self.hss_iter_min.setValidator(QIntValidator())
        self.hss_iter_max.setValidator(QIntValidator())
        self.hss_bs_min.setValidator(QIntValidator())
        self.hss_bs_max.setValidator(QIntValidator())
        self.hss_tol_min.setValidator(QDoubleValidator())
        self.hss_tol_max.setValidator(QDoubleValidator())
        self.pshl.setCurrentText('3')
        self.ps_hls_4.setDisabled(True)
        self.ps_hls_5.setDisabled(True)
        self.hss_hls_min_4.setDisabled(True)
        self.hss_hls_max_4.setDisabled(True)
        self.hss_hls_min_5.setDisabled(True)
        self.hss_hls_max_5.setDisabled(True)
        self.hsshl.setCurrentText('3')
        self.e = None

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
        model.model = MLPRegressor(
            hidden_layer_sizes=model.parameters['hidden_layer_sizes'],
            activation=model.parameters['activation'],
            solver=model.parameters['solver'],
            alpha=model.parameters['alpha'],
            batch_size=model.parameters['batch_size'],
            max_iter=model.parameters['max_iter'],
            tol=model.parameters['tol'],
            early_stopping=model.parameters['early_stopping'],
            validation_fraction=model.parameters['validation_fraction']
        )
        return model

    def define_model(self, selection):
        if selection == "yes":
            self.model.model_classes = "r"
            self.model.model_names_eng = "mlp"
            self.model.model_names = "Neural Network"
            opt_hl_list = []
            if eval(self.pshl.currentText()) == 1:
                try:
                    opt_hl_list.append(eval(self.ps_hls.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls.setText("8")
                    opt_hl_list.append(eval(self.ps_hls.text()))
            elif eval(self.pshl.currentText()) == 2:
                try:
                    opt_hl_list.append(eval(self.ps_hls.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls.setText("8")
                    opt_hl_list.append(eval(self.ps_hls.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_2.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
            elif eval(self.pshl.currentText()) == 3:
                try:
                    opt_hl_list.append(eval(self.ps_hls.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls.setText("8")
                    opt_hl_list.append(eval(self.ps_hls.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_2.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_3.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
            elif eval(self.pshl.currentText()) == 4:
                try:
                    opt_hl_list.append(eval(self.ps_hls.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls.setText("8")
                    opt_hl_list.append(eval(self.ps_hls.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_2.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_3.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_4.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_4.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_4.text()))
            else:
                try:
                    opt_hl_list.append(eval(self.ps_hls.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls.setText("8")
                    opt_hl_list.append(eval(self.ps_hls.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_2.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_2.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_3.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_3.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_4.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_4.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_4.text()))
                try:
                    opt_hl_list.append(eval(self.ps_hls_5.text()))
                except Exception as e:
                    self.e = str(e)
                    self.ps_hls_5.setText("8")
                    opt_hl_list.append(eval(self.ps_hls_5.text()))
            opt_hl_tuple = tuple(opt_hl_list)
            self.model.parameters['hidden_layer_sizes'] = opt_hl_tuple
            if self.ps_a_ident.isChecked():
                self.model.parameters['activation'] = 'identity'
            elif self.ps_a_tanh.isChecked():
                self.model.parameters['activation'] = 'logistic'
            elif self.ps_a_log.isChecked():
                self.model.parameters['activation'] = 'tanh'
            else:
                self.model.parameters['activation'] = 'relu'
            if self.ps_s_lbfgs.isChecked():
                self.model.parameters['solver'] = 'lbfgs'
            elif self.ps_s_sgd.isChecked():
                self.model.parameters['solver'] = 'sgd'
            else:
                self.model.parameters['solver'] = 'adam'
            try:
                self.model.parameters['batch_size'] = eval(self.ps_bs_m.text())
            except Exception as e:
                self.e = str(e)
                self.ps_bs_m.setText('auto')
                self.model.parameters['batch_size'] = self.ps_bs_m.text()
            try:
                self.model.parameters['alpha'] = eval(self.ps_alpha.text())
            except Exception as e:
                self.e = str(e)
                self.ps_alpha.setText("0.0001")
                self.model.parameters['alpha'] = eval(self.ps_alpha.text())
            try:
                self.model.parameters['tol'] = eval(self.ps_tol.text())
            except Exception as e:
                self.e = str(e)
                self.ps_tol.setText("0.0001")
                self.model.parameters['tol'] = eval(self.ps_tol.text())
            try:
                self.model.parameters['max_iter'] = eval(self.ps_iter.text())
            except Exception as e:
                self.e = str(e)
                self.ps_iter.setText("200")
                self.model.parameters['max_iter'] = eval(self.ps_iter.text())
            if self.groupBox_8.isChecked():
                self.model.parameters['early_stopping'] = True
            else:
                self.model.parameters['early_stopping'] = False
            try:
                self.model.parameters['validation_fraction'] = eval(self.ps_es_vsf.text())
            except Exception as e:
                self.e = str(e)
                self.ps_es_vsf.setText("0.1")
                self.model.parameters['validation_fraction'] = eval(self.ps_es_vsf.text())
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
                    self.e = str(e)
                    self.hss_times.setText("20")
                    self.model.opt_times = eval(self.hss_times.text())
                fold_cmbox_text = self.hss_fd.currentText()
                self.model.opt_folds = eval(fold_cmbox_text)
                opt_activation_list = []
                if self.hss_a_ident.isChecked():
                    opt_activation_list.append('identity')
                elif self.hss_a_log.isChecked():
                    opt_activation_list.append('logistic')
                elif self.hss_a_tanh.isChecked():
                    opt_activation_list.append('tanh')
                elif self.hss_a_relu.isChecked():
                    opt_activation_list.append('relu')
                if len(opt_activation_list) != 0:
                    self.model.opt_space['activation'] = hp.choice('activation', opt_activation_list)
                    self.model.opt_compare_space['activation'] = opt_activation_list
                opt_solver_list = []
                if self.hss_s_lbfgs.isChecked():
                    opt_solver_list.append('lbfgs')
                elif self.hss_s_sgd.isChecked():
                    opt_solver_list.append('sgd')
                elif self.hss_s_adam.isChecked():
                    opt_solver_list.append('adam')
                if len(opt_solver_list) != 0:
                    self.model.opt_space['solver'] = hp.choice('solver', opt_solver_list)
                    self.model.opt_compare_space['solver'] = opt_solver_list
                try:
                    self.model.opt_space['tol'] = hp.uniform('tol',
                                                             eval(self.hss_tol_min.text()),
                                                             eval(self.hss_tol_max.text()))
                    self.model.opt_valid_space['tol'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_tol_min.text()),
                                                       stop=eval(self.hss_tol_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = str(e)
                    self.hss_tol_min.setText("0.00001")
                    self.hss_tol_max.setText("0.01")
                    self.model.opt_space['tol'] = hp.uniform('tol',
                                                             eval(self.hss_tol_min.text()),
                                                             eval(self.hss_tol_max.text()))
                    self.model.opt_valid_space['tol'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_tol_min.text()),
                                                       stop=eval(self.hss_tol_max.text()),
                                                       num=10)]
                try:
                    self.model.opt_space['max_iter'] = hp.randint('max_iter',
                                                                  eval(self.hss_iter_min.text()),
                                                                  eval(self.hss_iter_max.text()))
                    self.model.opt_valid_space['max_iter'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_iter_min.text()),
                                                     stop=eval(self.hss_iter_max.text()),
                                                     num=10)]
                except Exception as e:
                    self.e = str(e)
                    self.hss_est_min.setText("20")
                    self.hss_est_max.setText("1000")
                    self.model.opt_space['max_iter'] = hp.randint('max_iter',
                                                                  eval(self.hss_iter_min.text()),
                                                                  eval(self.hss_iter_max.text()))
                    self.model.opt_valid_space['max_iter'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_iter_min.text()),
                                                     stop=eval(self.hss_iter_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['batch_size'] = hp.randint('batch_size',
                                                                    eval(self.hss_bs_min.text()),
                                                                    eval(self.hss_bs_max.text()))
                    self.model.opt_valid_space['batch_size'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_bs_min.text()),
                                                     stop=eval(self.hss_bs_max.text()),
                                                     num=10)]
                except Exception as e:
                    self.e = str(e)
                    self.hss_est_min.setText("2")
                    self.hss_est_max.setText("200")
                    self.model.opt_space['batch_size'] = hp.randint('batch_size',
                                                                    eval(self.hss_est_min.text()),
                                                                    eval(self.hss_est_max.text()))
                    self.model.opt_valid_space['batch_size'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_bs_min.text()),
                                                     stop=eval(self.hss_bs_max.text()),
                                                     num=10)]
                opt_hl_final_list = []
                opt_hl_min_list = []
                opt_hl_max_list = []
                if eval(self.hsshl.currentText()) == 1:
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls.setText("4")
                        self.ps_hls.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    for i in range(opt_hl_min_list[0], opt_hl_max_list[0] + 1):
                        temp_list = [i]
                        temp_tuple = tuple(temp_list)
                        opt_hl_final_list.append(temp_tuple)
                    self.model.opt_space['hidden_layer_sizes'] = hp.choice('hidden_layer_sizes', opt_hl_final_list)
                elif eval(self.hsshl.currentText()) == 2:
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls.setText("4")
                        self.ps_hls.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_2.setText("4")
                        self.ps_hls_2.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    for i in range(opt_hl_min_list[0], opt_hl_max_list[0] + 1):
                        for j in range(opt_hl_min_list[1], opt_hl_max_list[1] + 1):
                            temp_list = [i, j]
                            temp_tuple = tuple(temp_list)
                            opt_hl_final_list.append(temp_tuple)
                    self.model.opt_space['hidden_layer_sizes'] = hp.choice('hidden_layer_sizes', opt_hl_final_list)
                elif eval(self.hsshl.currentText()) == 3:
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls.setText("4")
                        self.ps_hls.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_2.setText("4")
                        self.ps_hls_2.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_3.setText("4")
                        self.ps_hls_3.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    for i in range(opt_hl_min_list[0], opt_hl_max_list[0] + 1):
                        for j in range(opt_hl_min_list[1], opt_hl_max_list[1] + 1):
                            for k in range(opt_hl_min_list[2], opt_hl_max_list[2] + 1):
                                temp_list = [i, j, k]
                                temp_tuple = tuple(temp_list)
                                opt_hl_final_list.append(temp_tuple)
                    self.model.opt_space['hidden_layer_sizes'] = hp.choice('hidden_layer_sizes', opt_hl_final_list)
                elif eval(self.hsshl.currentText()) == 4:
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls.setText("4")
                        self.ps_hls.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_2.setText("4")
                        self.ps_hls_2.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_3.setText("4")
                        self.ps_hls_3.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_4.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_4.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_4.setText("4")
                        self.ps_hls_4.setText("8")
                        opt_hl_min_list.append(eval(self.hss_hls_min_4.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_4.text()))
                    for i in range(opt_hl_min_list[0], opt_hl_max_list[0] + 1):
                        for j in range(opt_hl_min_list[1], opt_hl_max_list[1] + 1):
                            for k in range(opt_hl_min_list[2], opt_hl_max_list[2] + 1):
                                for m in range(opt_hl_min_list[3], opt_hl_max_list[3] + 1):
                                    temp_list = [i, j, k, m]
                                    temp_tuple = tuple(temp_list)
                                    opt_hl_final_list.append(temp_tuple)
                else:
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls.setText("4")
                        self.ps_hls.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_2.setText("4")
                        self.ps_hls_2.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_2.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_2.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_3.setText("4")
                        self.ps_hls_3.setText("16")
                        opt_hl_min_list.append(eval(self.hss_hls_min_3.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_3.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_4.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_4.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_4.setText("4")
                        self.ps_hls_4.setText("8")
                        opt_hl_min_list.append(eval(self.hss_hls_min_4.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_4.text()))
                    try:
                        opt_hl_min_list.append(eval(self.hss_hls_min_5.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_5.text()))
                    except Exception as e:
                        self.e = str(e)
                        self.ps_hls_5.setText("2")
                        self.ps_hls_5.setText("4")
                        opt_hl_min_list.append(eval(self.hss_hls_min_5.text()))
                        opt_hl_max_list.append(eval(self.hss_hls_max_5.text()))
                    for i in range(opt_hl_min_list[0], opt_hl_max_list[0] + 1):
                        for j in range(opt_hl_min_list[1], opt_hl_max_list[1] + 1):
                            for k in range(opt_hl_min_list[2], opt_hl_max_list[2] + 1):
                                for m in range(opt_hl_min_list[3], opt_hl_max_list[3] + 1):
                                    for n in range(opt_hl_min_list[3], opt_hl_max_list[3] + 1):
                                        temp_list = [i, j, k, m, n]
                                        temp_tuple = tuple(temp_list)
                                        opt_hl_final_list.append(temp_tuple)
            else:
                self.model.opt_selection = False
            try:
                self.model.model = MLPRegressor(
                    hidden_layer_sizes=self.model.parameters['hidden_layer_sizes'],
                    activation=self.model.parameters['activation'],
                    solver=self.model.parameters['solver'],
                    alpha=self.model.parameters['alpha'],
                    batch_size=self.model.parameters['batch_size'],
                    max_iter=self.model.parameters['max_iter'],
                    tol=self.model.parameters['tol'],
                    early_stopping=self.model.parameters['early_stopping'],
                    validation_fraction=self.model.parameters['validation_fraction']
                )
                self.model_signal[object].emit(self.model)
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pass

    @pyqtSlot(str)
    def on_pshl_activated(self, value):
        if eval(value) == 1:
            self.ps_hls.setEnabled(True)
            self.ps_hls_2.setDisabled(True)
            self.ps_hls_3.setDisabled(True)
            self.ps_hls_4.setDisabled(True)
            self.ps_hls_5.setDisabled(True)
        elif eval(value) == 2:
            self.ps_hls.setEnabled(True)
            self.ps_hls_2.setEnabled(True)
            self.ps_hls_3.setDisabled(True)
            self.ps_hls_4.setDisabled(True)
            self.ps_hls_5.setDisabled(True)
        elif eval(value) == 3:
            self.ps_hls.setEnabled(True)
            self.ps_hls_2.setEnabled(True)
            self.ps_hls_3.setEnabled(True)
            self.ps_hls_4.setDisabled(True)
            self.ps_hls_5.setDisabled(True)
        elif eval(value) == 4:
            self.ps_hls.setEnabled(True)
            self.ps_hls_2.setEnabled(True)
            self.ps_hls_3.setEnabled(True)
            self.ps_hls_4.setEnabled(True)
            self.ps_hls_5.setDisabled(True)
        else:
            self.ps_hls.setEnabled(True)
            self.ps_hls_2.setEnabled(True)
            self.ps_hls_3.setEnabled(True)
            self.ps_hls_4.setEnabled(True)
            self.ps_hls_5.setEnabled(True)

    @pyqtSlot(str)
    def on_hsshl_activated(self, value):
        if eval(value) == 1:
            self.hss_hls_min.setEnabled(True)
            self.hss_hls_max.setEnabled(True)
            self.hss_hls_min_2.setDisabled(True)
            self.hss_hls_max_2.setDisabled(True)
            self.hss_hls_min_3.setDisabled(True)
            self.hss_hls_min_3.setDisabled(True)
            self.hss_hls_min_4.setDisabled(True)
            self.hss_hls_max_4.setDisabled(True)
            self.hss_hls_min_5.setDisabled(True)
            self.hss_hls_max_5.setDisabled(True)
        elif eval(value) == 2:
            self.hss_hls_min.setEnabled(True)
            self.hss_hls_max.setEnabled(True)
            self.hss_hls_min_2.setEnabled(True)
            self.hss_hls_max_2.setEnabled(True)
            self.hss_hls_min_3.setDisabled(True)
            self.hss_hls_min_3.setDisabled(True)
            self.hss_hls_min_4.setDisabled(True)
            self.hss_hls_max_4.setDisabled(True)
            self.hss_hls_min_5.setDisabled(True)
            self.hss_hls_max_5.setDisabled(True)
        elif eval(value) == 3:
            self.hss_hls_min.setEnabled(True)
            self.hss_hls_max.setEnabled(True)
            self.hss_hls_min_2.setEnabled(True)
            self.hss_hls_max_2.setEnabled(True)
            self.hss_hls_min_3.setEnabled(True)
            self.hss_hls_max_3.setEnabled(True)
            self.hss_hls_min_4.setDisabled(True)
            self.hss_hls_max_4.setDisabled(True)
            self.hss_hls_min_5.setDisabled(True)
            self.hss_hls_max_5.setDisabled(True)
        elif eval(value) == 4:
            self.hss_hls_min.setEnabled(True)
            self.hss_hls_max.setEnabled(True)
            self.hss_hls_min_2.setEnabled(True)
            self.hss_hls_max_2.setEnabled(True)
            self.hss_hls_min_3.setEnabled(True)
            self.hss_hls_max_3.setEnabled(True)
            self.hss_hls_min_4.setEnabled(True)
            self.hss_hls_max_4.setEnabled(True)
            self.hss_hls_min_5.setDisabled(True)
            self.hss_hls_max_5.setDisabled(True)
        else:
            self.hss_hls_min.setEnabled(True)
            self.hss_hls_max.setEnabled(True)
            self.hss_hls_min_2.setEnabled(True)
            self.hss_hls_max_2.setEnabled(True)
            self.hss_hls_min_3.setEnabled(True)
            self.hss_hls_max_3.setEnabled(True)
            self.hss_hls_min_4.setEnabled(True)
            self.hss_hls_max_4.setEnabled(True)
            self.hss_hls_min_5.setEnabled(True)
            self.hss_hls_max_5.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_nn()
    win.show()
    sys.exit(app.exec_())
