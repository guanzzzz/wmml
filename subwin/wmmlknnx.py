import sys
import numpy as np
from subwin.wmmlknn import Ui_WMML_knn
from MLparameters import *
from hyperopt import hp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


class WMML_knn(QWidget, Ui_WMML_knn):
    """
    WMML K-nearest Neighbor Model Define
    Output signal:
    model: Model object with parameter defined neutral network model
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_knn, self).__init__()
        self.setupUi(self)
        self.model = Model()
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.ps_n.setValidator(QIntValidator())
        self.hss_n_min.setValidator(QIntValidator())
        self.hss_n_max.setValidator(QIntValidator())
        self.hss_times.setValidator(QIntValidator())
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
        model.model = KNeighborsRegressor(
            n_neighbors=model.parameters['n_neighbors'],
            weights=model.parameters['weights'],
            algorithm=model.parameters['algorithm'],
            metric=model.parameters['metric']
        )
        return model

    def define_model(self, selection):
        if selection == "yes":
            self.model.model_classes = "r"
            self.model.model_names = "K-Neighbors"
            self.model.model_names_eng = "knn"
            try:
                self.model.parameters['n_neighbors'] = eval(self.ps_n.text())
            except Exception as e:
                self.e = str(e)
                self.ps_est.setText("5")
                self.model.parameters['n_neighbors'] = eval(self.ps_n.text())
            if self.ps_a_auto.isChecked():
                self.model.parameters['algorithm'] = 'auto'
            elif self.ps_a_bt.isChecked():
                self.model.parameters['algorithm'] = 'ball_tree'
            elif self.ps_a_kt.isChecked():
                self.model.parameters['algorithm'] = 'kd_tree'
            else:
                self.model.parameters['algorithm'] = 'brute'
            if self.ps_m_eucl.isChecked():
                self.model.parameters['metric'] = 'euclidean'
            elif self.ps_m_manh.isChecked():
                self.model.parameters['metric'] = 'manhattan'
            elif self.ps_m_cheb.isChecked():
                self.model.parameters['metric'] = 'chebyshev'
            elif self.ps_m_mink.isChecked():
                self.model.parameters['metric'] = 'minkowski'
            elif self.ps_m_wink.isChecked():
                self.model.parameters['metric'] = 'wminkowski'
            elif self.ps_m_seuc.isChecked():
                self.model.parameters['metric'] = 'seuclidean'
            else:
                self.model.parameters['metric'] = 'mahalanobis'
            if self.ps_w_uni.isChecked():
                self.model.parameters['weights'] = 'uniform'
            else:
                self.model.parameters['weights'] = 'distance'
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
                opt_algorithm_list = []
                if self.hss_a_auto.isChecked():
                    opt_algorithm_list.append('auto')
                elif self.hss_a_bt.isChecked():
                    opt_algorithm_list.append('ball_tree')
                elif self.hss_a_kt.isChecked():
                    opt_algorithm_list.append('kd_tree')
                elif self.hss_a_bru.isChecked():
                    opt_algorithm_list.append('brute')
                if len(opt_algorithm_list) != 0:
                    self.model.opt_space['algorithm'] = hp.choice('algorithm', opt_algorithm_list)
                    self.model.opt_compare_space['algorithm'] = opt_algorithm_list
                opt_weights_list = []
                if self.hss_a_auto.isChecked():
                    opt_weights_list.append('uniform')
                elif self.hss_a_bt.isChecked():
                    opt_weights_list.append('distance')
                if len(opt_weights_list) != 0:
                    self.model.opt_space['weights'] = hp.choice('weights', opt_weights_list)
                    self.model.opt_compare_space['weights'] = opt_weights_list
                opt_metric_list = []
                if self.hss_m_euc.isChecked():
                    opt_metric_list.append('euclidean')
                elif self.hss_m_manh.isChecked():
                    opt_metric_list.append('manhattan')
                elif self.hss_m_cheb.isChecked():
                    opt_metric_list.append('chebyshev')
                elif self.hss_m_mink.isChecked():
                    opt_metric_list.append('minkowski')
                elif self.hss_m_wink.isChecked():
                    opt_metric_list.append('wminkowski')
                elif self.hss_m_seuc.isChecked():
                    opt_metric_list.append('seuclidean')
                elif self.hss_m_maha.isChecked():
                    opt_metric_list.append('mahalanobis')
                if len(opt_metric_list) != 0:
                    self.model.opt_space['metric'] = hp.choice('metric', opt_metric_list)
                    self.model.opt_compare_space['metric'] = opt_metric_list
                try:
                    self.model.opt_space['n_neighbors'] = hp.randint('n_neighbors',
                                                                     eval(self.hss_n_min.text()),
                                                                     eval(self.hss_n_max.text()))
                    self.model.opt_valid_space['n_neighbors'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_n_min.text()),
                                                     stop=eval(self.hss_n_max.text()),
                                                     num=10)]
                except Exception as e:
                    self.e = str(e)
                    self.hss_n_min.setText("2")
                    self.hss_n_max.setText("15")
                    self.model.opt_space['n_neighbors'] = hp.randint('n_neighbors',
                                                                     eval(self.hss_n_min.text()),
                                                                     eval(self.hss_n_max.text()))
                    self.model.opt_valid_space['n_neighbors'] = \
                        [int(x) for x in np.linspace(start=eval(self.hss_n_min.text()),
                                                     stop=eval(self.hss_n_max.text()),
                                                     num=10)]
            else:
                self.model.opt_selection = False
            try:
                self.model.model = KNeighborsRegressor(
                    n_neighbors=self.model.parameters['n_neighbors'],
                    weights=self.model.parameters['weights'],
                    algorithm=self.model.parameters['algorithm'],
                    metric=self.model.parameters['metric']
                )
                self.model_signal[object].emit(self.model)
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_knn()
    win.show()
    sys.exit(app.exec_())
