import sys
import numpy as np
from subwin.wmmlgbdt import Ui_WMML_gbdt
from subwin.wmmlsubeng import options
from MLparameters import *
from hyperopt import hp
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class WMML_gbdt(QWidget, Ui_WMML_gbdt):
    """
    WMML Neutral Network Model Define
    Output signal:
    model: Model object with parameter defined neutral network model
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_gbdt, self).__init__()
        self.setupUi(self)
        self.model = Model()
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.l1_ratio_validator = QDoubleValidator()
        self.l1_ratio_validator.setRange(0.0, 1.0)
        self.l1_ratio_validator.setNotation(QDoubleValidator.StandardNotation)
        self.temp_model_fin = self.e = None
        self.finmd_list = [('None', None)]
        self.finmd_name_list = ['Dummy Estimator']
        self.f_load_f1_2.append("Init Model Class:{value}".format(value=self.finmd_list[0][1]))
        self.f_load_f1.append("Init Model:{value}".format(value=self.finmd_name_list[0]))
        self.f_load.clicked.connect(self.ld_finmd)
        self.addButton_2.clicked.connect(self.ad_finmd)
        self.addButton_3.clicked.connect(self.rc_finmd)
        self.ps_lr.setValidator(QDoubleValidator())
        self.ps_est.setValidator(QIntValidator())
        self.ps_ss.setValidator(self.l1_ratio_validator)
        self.ps_mss.setValidator(QIntValidator())
        self.ps_msl.setValidator(QIntValidator())
        self.ps_mwfl.setValidator(QDoubleValidator())
        self.ps_md.setValidator(QIntValidator())
        self.ps_mid.setValidator(QDoubleValidator())
        self.ps_rand.setValidator(QIntValidator())
        self.ps_alpha.setValidator(self.l1_ratio_validator)
        self.ps_mln.setValidator(QIntValidator())
        self.ps_tol.setValidator(QDoubleValidator())
        self.hss_times.setValidator(QIntValidator())
        self.hss_lr_min.setValidator(QDoubleValidator())
        self.hss_lr_max.setValidator(QDoubleValidator())
        self.hss_est_min.setValidator(QIntValidator())
        self.hss_est_max.setValidator(QIntValidator())
        self.hss_ss_min.setValidator(self.l1_ratio_validator)
        self.hss_ss_max.setValidator(self.l1_ratio_validator)
        self.hss_mss_min.setValidator(QIntValidator())
        self.hss_mss_max.setValidator(QIntValidator())
        self.hss_msl_min.setValidator(QIntValidator())
        self.hss_msl_max.setValidator(QIntValidator())
        self.hss_mwfl_min.setValidator(QDoubleValidator())
        self.hss_mwfl_max.setValidator(QDoubleValidator())
        self.hss_md_min.setValidator(QIntValidator())
        self.hss_md_max.setValidator(QIntValidator())
        self.hss_mid_min.setValidator(QDoubleValidator())
        self.hss_mid_max.setValidator(QDoubleValidator())
        self.hss_alpha_min.setValidator(self.l1_ratio_validator)
        self.hss_alpha_max.setValidator(self.l1_ratio_validator)
        self.hss_mln_min.setValidator(QIntValidator())
        self.hss_mln_max.setValidator(QIntValidator())
        self.hss_tol_min.setValidator(QDoubleValidator())
        self.hss_tol_max.setValidator(QDoubleValidator())

    def ld_finmd(self):
        try:
            fbx = QFileDialog()
            data_position = QFileDialog.getOpenFileName(fbx, 'select file', '', 'pkcls files(*.pkcls)', options=options)
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
            self.e = e
            self.f_load_info.append("Model Loaded Failed")

    # noinspection PyTypeChecker
    def ad_finmd(self):
        try:
            self.finmd_name_list[0] = self.temp_model_fin.model_names
            temp_model_list = [self.temp_model_fin.model_names_eng, self.temp_model_fin.model]
            self.finmd_list.pop()
            self.finmd_list.append(tuple(temp_model_list))
            self.f_load_f1_2.clear()
            self.f_load_f1.clear()
            self.f_load_f1_2.append("Init Model Class:{value}".format(value=self.finmd_list[0][1]))
            self.f_load_f1.append("Init Model:{value}".format(value=self.finmd_name_list[0]))
            self.f_load_info.append("Init Model Updated")
        except Exception as e:
            self.e = e
            self.f_load_info.append("Updated Failed")

    def rc_finmd(self):
        try:
            self.finmd_list = [('None', None)]
            self.finmd_name_list = ['Simple Estimator']
            self.f_load_f1_2.clear()
            self.f_load_f1.clear()
            self.f_load_f1_2.append("Init Model Class:{value}".format(value=self.finmd_list[0][1]))
            self.f_load_f1.append("Init Model:{value}".format(value=self.finmd_name_list[0]))
            self.f_load_info.append("Init Model Updated")
        except Exception as e:
            self.e = e
            self.f_load_info.append("Updated Failed")

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
        model.model = GradientBoostingRegressor(
            loss=model.parameters['loss'],
            learning_rate=model.parameters['learning_rate'],
            n_estimators=model.parameters['n_estimators'],
            subsample=model.parameters['subsample'],
            criterion=model.parameters['criterion'],
            min_samples_split=model.parameters['min_samples_split'],
            min_samples_leaf=model.parameters['min_samples_leaf'],
            min_weight_fraction_leaf=model.parameters['min_weight_fraction_leaf'],
            max_depth=model.parameters['max_depth'],
            min_impurity_decrease=model.parameters['min_impurity_decrease'],
            init=model.parameters['init'],
            random_state=model.parameters['random_state'],
            max_features=model.parameters['max_features'],
            alpha=model.parameters['alpha'],
            max_leaf_nodes=model.parameters['max_leaf_nodes'],
            tol=model.parameters['tol']
        )
        return model

    def define_model(self, selection):
        if selection == "yes":
            self.model.model_classes = "r"
            self.model.model_names = "Gradient Boosting"
            self.model.model_names_eng = "gbdtr"
            # 模型参数
            self.model.parameters['init'] = self.finmd_list[0][1]
            if self.ps_l_sqr.isChecked():
                self.model.parameters['loss'] = 'ls'
            elif self.ps_l_abr.isChecked():
                self.model.parameters['loss'] = 'lad'
            elif self.ps_l_hub.isChecked():
                self.model.parameters['loss'] = 'huber'
            else:
                self.model.parameters['loss'] = 'quantile'
            if self.ps_c_f.isChecked():
                self.model.parameters['criterion'] = 'friedman_mse'
            elif self.ps_c_sqr.isChecked():
                self.model.parameters['criterion'] = 'squared_error'
            elif self.ps_c_mse.isChecked():
                self.model.parameters['criterion'] = 'mse'
            else:
                self.model.parameters['criterion'] = 'mae'
            if self.ps_mf_n.isChecked():
                self.model.parameters['max_features'] = None
            elif self.ps_mf_auto.isChecked():
                self.model.parameters['max_features'] = 'auto'
            elif self.ps_mf_sqr.isChecked():
                self.model.parameters['max_features'] = 'sqrt'
            else:
                self.model.parameters['max_features'] = 'log2'
            try:
                self.model.parameters['learning_rate'] = eval(self.ps_lr.text())
            except Exception as e:
                self.e = e
                self.ps_lr.setText("0.1")
                self.model.parameters['learning_rate'] = eval(self.ps_lr.text())
            try:
                self.model.parameters['n_estimators'] = eval(self.ps_est.text())
            except Exception as e:
                self.e = e
                self.ps_est.setText("100")
                self.model.parameters['n_estimators'] = eval(self.ps_est.text())
            try:
                self.model.parameters['subsample'] = eval(self.ps_ss.text())
            except Exception as e:
                self.e = e
                self.ps_ss.setText("1.0")
                self.model.parameters['subsample'] = eval(self.ps_ss.text())
            try:
                self.model.parameters['min_samples_split'] = eval(self.ps_mss.text())
            except Exception as e:
                self.e = e
                self.ps_mss.setText("2")
                self.model.parameters['min_samples_split'] = eval(self.ps_mss.text())
            try:
                self.model.parameters['min_samples_leaf'] = eval(self.ps_msl.text())
            except Exception as e:
                self.e = e
                self.ps_msl.setText("1")
                self.model.parameters['min_samples_leaf'] = eval(self.ps_msl.text())
            try:
                self.model.parameters['min_weight_fraction_leaf'] = eval(self.ps_mwfl.text())
            except Exception as e:
                self.e = e
                self.ps_mwfl.setText("0.0000")
                self.model.parameters['min_weight_fraction_leaf'] = eval(self.ps_mwfl.text())
            try:
                self.model.parameters['max_depth'] = eval(self.ps_md.text())
            except Exception as e:
                self.e = e
                self.ps_md.setText("3")
                self.model.parameters['max_depth'] = eval(self.ps_md.text())
            try:
                self.model.parameters['min_impurity_decrease'] = eval(self.ps_mid.text())
            except Exception as e:
                self.e = e
                self.ps_mid.setText("0.0")
                self.model.parameters['min_impurity_decrease'] = eval(self.ps_mid.text())
            try:
                self.model.parameters['random_state'] = eval(self.ps_rand.text())
            except Exception as e:
                self.e = e
                self.ps_rand.setText('None')
                self.model.parameters['random_state'] = eval(self.ps_rand.text())
            try:
                self.model.parameters['alpha'] = eval(self.ps_alpha.text())
            except Exception as e:
                self.e = e
                self.ps_alpha.setText("0.9")
                self.model.parameters['alpha'] = eval(self.ps_alpha.text())
            try:
                self.model.parameters['max_leaf_nodes'] = eval(self.ps_mln.text())
            except Exception as e:
                self.e = e
                self.ps_mln.setText("None")
                self.model.parameters['max_leaf_nodes'] = eval(self.ps_mln.text())
            try:
                self.model.parameters['tol'] = eval(self.ps_tol.text())
            except Exception as e:
                self.e = e
                self.ps_tol.setText("0.0001")
                self.model.parameters['tol'] = eval(self.ps_tol.text())
            # 超参数优化空间
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
                    self.e = e
                    self.hss_times.setText("20")
                    self.model.opt_times = eval(self.hss_times.text())
                fold_cmbox_text = self.hss_fd.currentText()
                self.model.opt_folds = eval(fold_cmbox_text)
                temp_default_bsemd = None
                opt_base_estimator_list = [temp_default_bsemd, self.finmd_list[0][1]]
                self.model.opt_space['init'] = hp.choice('init', opt_base_estimator_list)
                self.model.opt_compare_space['init'] = opt_base_estimator_list
                opt_loss_list = []
                if self.hss_l_sqr.isChecked():
                    opt_loss_list.append('ls')
                elif self.hss_l_abr.isChecked():
                    opt_loss_list.append('lad')
                elif self.hss_l_hub.isChecked():
                    opt_loss_list.append('huber')
                elif self.hss_l_quat.isChecked():
                    opt_loss_list.append('quantile')
                if len(opt_loss_list) != 0:
                    self.model.opt_space['loss'] = hp.choice('loss', opt_loss_list)
                    self.model.opt_compare_space['loss'] = opt_loss_list
                opt_criterion_list = []
                if self.hss_c_f.isChecked():
                    opt_criterion_list.append('friedman_mse')
                elif self.hss_c_sqr.isChecked():
                    opt_criterion_list.append('squared_error')
                elif self.hss_c_mse.isChecked():
                    opt_criterion_list.append('mse')
                elif self.hss_c_mae.isChecked():
                    opt_criterion_list.append('mae')
                if len(opt_criterion_list) != 0:
                    self.model.opt_space['criterion'] = hp.choice('criterion', opt_criterion_list)
                    self.model.opt_compare_space['criterion'] = opt_criterion_list
                opt_max_features_list = []
                if self.hss_mf_n.isChecked():
                    opt_max_features_list.append(None)
                elif self.hss_mf_auto.isChecked():
                    opt_max_features_list.append('auto')
                elif self.hss_mf_sqr.isChecked():
                    opt_max_features_list.append('sqrt')
                elif self.hss_mf_log2.isChecked():
                    opt_max_features_list.append('log2')
                if len(opt_max_features_list) != 0:
                    self.model.opt_space['max_features'] = hp.choice('max_features', opt_max_features_list)
                    self.model.opt_compare_space['max_features'] = opt_max_features_list
                try:
                    self.model.opt_space['learning_rate'] = hp.uniform('learning_rate',
                                                                       eval(self.hss_lr_min.text()),
                                                                       eval(self.hss_lr_max.text()))
                    self.model.opt_valid_space['learning_rate'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_lr_min.text()),
                                                       stop=eval(self.hss_lr_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = e
                    self.hss_lr_min.setText("0.01")
                    self.hss_lr_max.setText("1")
                    self.model.opt_space['learning_rate'] = hp.uniform('learning_rate',
                                                                       eval(self.hss_lr_min.text()),
                                                                       eval(self.hss_lr_max.text()))
                    self.model.opt_valid_space['learning_rate'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_lr_min.text()),
                                                       stop=eval(self.hss_lr_max.text()),
                                                       num=10)]
                try:
                    self.model.opt_space['n_estimators'] = hp.randint('n_estimators',
                                                                      eval(self.hss_est_min.text()),
                                                                      eval(self.hss_est_max.text()))
                    self.model.opt_valid_space['n_estimators'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_est_min.text()),
                                                     stop=eval(self.hss_est_max.text()),
                                                     num=10)]
                except Exception as e:
                    self.e = e
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
                    self.model.opt_space['subsample'] = hp.uniform('subsample',
                                                                   eval(self.hss_ss_min.text()),
                                                                   eval(self.hss_ss_max.text()))
                    self.model.opt_valid_space['subsample'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_ss_min.text()),
                                                       stop=eval(self.hss_ss_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = e
                    self.hss_ss_min.setText("0.5")
                    self.hss_ss_max.setText("1")
                    self.model.opt_space['subsample'] = hp.uniform('subsample',
                                                                   eval(self.hss_ss_min.text()),
                                                                   eval(self.hss_ss_max.text()))
                    self.model.opt_valid_space['subsample'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_ss_min.text()),
                                                       stop=eval(self.hss_ss_max.text()),
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
                    self.e = e
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
                    self.e = e
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
                    self.e = e
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
                    self.model.opt_space['max_depth'] = hp.randint('max_depth',
                                                                   eval(self.hss_md_min.text()),
                                                                   eval(self.hss_md_max.text()))
                    self.model.opt_valid_space['max_depth'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_md_min.text()),
                                                     stop=eval(self.hss_md_max.text()),
                                                     num=10)]
                except Exception as e:
                    self.e = e
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
                    self.model.opt_space['min_impurity_decrease'] = hp.uniform('min_impurity_decrease',
                                                                               eval(self.hss_mid_min.text()),
                                                                               eval(self.hss_mid_max.text()))
                    self.model.opt_valid_space['min_impurity_decrease'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mid_min.text()),
                                                       stop=eval(self.hss_mid_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = e
                    self.hss_mid_min.setText("0.0000")
                    self.hss_mid_max.setText("0.0001")
                    self.model.opt_space['min_impurity_decrease'] = hp.uniform('min_impurity_decrease',
                                                                               eval(self.hss_mid_min.text()),
                                                                               eval(self.hss_mid_max.text()))
                    self.model.opt_valid_space['min_impurity_decrease'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_mid_min.text()),
                                                       stop=eval(self.hss_mid_max.text()),
                                                       num=10)]
                try:
                    self.model.opt_space['alpha'] = hp.uniform('alpha',
                                                               eval(self.hss_alpha_min.text()),
                                                               eval(self.hss_alpha_max.text()))
                    self.model.opt_valid_space['alpha'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_alpha_min.text()),
                                                       stop=eval(self.hss_alpha_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = e
                    self.hss_alpha_min.setText("0.2")
                    self.hss_alpha_max.setText("1.0")
                    self.model.opt_space['alpha'] = hp.uniform('alpha',
                                                               eval(self.hss_alpha_min.text()),
                                                               eval(self.hss_alpha_max.text()))
                    self.model.opt_valid_space['alpha'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_alpha_min.text()),
                                                       stop=eval(self.hss_alpha_max.text()),
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
                    self.e = e
                    self.hss_mln_min.setText("0.1")
                    self.hss_mln_max.setText("10")
                    self.model.opt_space['max_leaf_nodes'] = hp.randint('max_leaf_nodes',
                                                                        eval(self.hss_mln_min.text()),
                                                                        eval(self.hss_mln_max.text()))
                    self.model.opt_valid_space['max_leaf_nodes'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_mln_min.text()),
                                                     stop=eval(self.hss_mln_max.text()),
                                                     num=10)]
                try:
                    self.model.opt_space['tol'] = hp.uniform('tol',
                                                             eval(self.hss_tol_min.text()),
                                                             eval(self.hss_tol_max.text()))
                    self.model.opt_valid_space['tol'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_tol_min.text()),
                                                       stop=eval(self.hss_tol_max.text()),
                                                       num=10)]
                except Exception as e:
                    self.e = e
                    self.hss_tol_min.setText("0.0001")
                    self.hss_tol_max.setText("0.001")
                    self.model.opt_space['tol'] = hp.uniform('tol',
                                                             eval(self.hss_tol_min.text()),
                                                             eval(self.hss_tol_max.text()))
                    self.model.opt_valid_space['tol'] = \
                        [float(x) for x in np.linspace(start=eval(self.hss_tol_min.text()),
                                                       stop=eval(self.hss_tol_max.text()),
                                                       num=10)]
            else:
                self.model.opt_selection = False
            try:
                self.model.model = GradientBoostingRegressor(
                    loss=self.model.parameters['loss'],
                    learning_rate=self.model.parameters['learning_rate'],
                    n_estimators=self.model.parameters['n_estimators'],
                    subsample=self.model.parameters['subsample'],
                    criterion=self.model.parameters['criterion'],
                    min_samples_split=self.model.parameters['min_samples_split'],
                    min_samples_leaf=self.model.parameters['min_samples_leaf'],
                    min_weight_fraction_leaf=self.model.parameters['min_weight_fraction_leaf'],
                    max_depth=self.model.parameters['max_depth'],
                    min_impurity_decrease=self.model.parameters['min_impurity_decrease'],
                    init=self.model.parameters['init'],
                    random_state=self.model.parameters['random_state'],
                    max_features=self.model.parameters['max_features'],
                    alpha=self.model.parameters['alpha'],
                    max_leaf_nodes=self.model.parameters['max_leaf_nodes'],
                    tol=self.model.parameters['tol']
                )
                self.model_signal[object].emit(self.model)
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pass

    @pyqtSlot(str)
    def on_ps_ss_textChanged(self, value):
        try:
            if float(value) > 1:
                self.ps_ss.clear()
                self.ps_ss.setText("1.00")
        except Exception as e:
            self.e = e

    @pyqtSlot(str)
    def on_ps_alpha_textChanged(self, value):
        try:
            if float(value) > 1:
                self.ps_alpha.clear()
                self.ps_alpha.setText("1.00")
        except Exception as e:
            self.e = e

    @pyqtSlot(str)
    def on_hss_ss_min_textChanged(self, value):
        try:
            if float(value) > 1:
                self.hss_ss_min.clear()
                self.hss_ss_min.setText("1.00")
        except Exception as e:
            self.e = e

    @pyqtSlot(str)
    def on_hss_ss_max_textChanged(self, value):
        try:
            if float(value) > 1:
                self.hss_ss_max.clear()
                self.hss_ss_max.setText("1.00")
        except Exception as e:
            self.e = e

    @pyqtSlot(str)
    def on_hss_alpha_min_textChanged(self, value):
        try:
            if float(value) > 1:
                self.hss_alpha_min.clear()
                self.hss_alpha_min.setText("1.00")
        except Exception as e:
            self.e = e

    @pyqtSlot(str)
    def on_hss_alpha_max_textChanged(self, value):
        try:
            if float(value) > 1:
                self.hss_alpha_max.clear()
                self.hss_alpha_max.setText("1.00")
        except Exception as e:
            self.e = e


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_gbdt()
    win.show()
    sys.exit(app.exec_())
