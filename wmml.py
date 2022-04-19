from subwin.wmmlmain import Ui_WMML_mainwindow
from subwin.wmmlcsvx import WMML_csv
from subwin.wmmlprepx import WMML_prep
from subwin.wmmlcorrelationx import WMML_correlation
from subwin.wmmlpcax import WMML_pca
from subwin.wmmlstkx import WMML_stk
from subwin.wmmlrfx import WMML_rf
from subwin.wmmlknnx import WMML_knn
from subwin.wmmlgbdtx import WMML_gbdt
from subwin.wmmlnetx import WMML_nn
from subwin.wmmlsrx import WMML_sr
from subwin.wmmltestandscorex import WMML_tas
from subwin.wmmlpredictionx import WMML_pred
from subwin.wmmlsubeng import options
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import sys
import warnings
warnings.filterwarnings('ignore')


class WMML_Main(Ui_WMML_mainwindow, QtWidgets.QMainWindow):
    """
    Model Main Widget
    available signal:
    data: Dataset with Features, Metas and Targets
    model: Model object with  ML model and relative information
    """
    receiver_signal_csv_widget = QtCore.pyqtSignal(object)
    receiver_signal_prep_widget = QtCore.pyqtSignal(object)
    receiver_signal_pca_widget = QtCore.pyqtSignal(object)
    receiver_signal_ml_model = QtCore.pyqtSignal(object)

    def __init__(self):
        super(WMML_Main, self).__init__()
        self.setupUi(self)
        self.available_subwin_dict = {}
        self.available_mdwin_dict = {}
        self.e = None

    def send_potential_receiver(self, source_widget_name):
        all_activate_widget_list = [i for i in self.available_subwin_dict.keys()]
        all_activate_widget_list.remove(source_widget_name)
        if source_widget_name == 'csv_widget':
            self.receiver_signal_csv_widget.emit(all_activate_widget_list)
        elif source_widget_name == 'prep_widget':
            self.receiver_signal_prep_widget.emit(all_activate_widget_list)
        elif source_widget_name == 'pca_widget':
            self.receiver_signal_pca_widget.emit(all_activate_widget_list)

    @QtCore.pyqtSlot()
    def on_jz_ld_btn_clicked(self):
        try:
            if self.available_subwin_dict['csv_widget'].isVisible():
                self.available_subwin_dict['csv_widget'].show()
            else:
                self.available_subwin_dict['csv_widget'].showMaximized()
        except Exception as e:
            self.e = e
            file_diag = QtWidgets.QFileDialog()
            csv_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'select file', '', 'Csv Data files(*.csv)',
                                                                 options=options)
            if isinstance(csv_position, tuple):
                if csv_position[0] == '':
                    pass
                else:
                    path_data_position = csv_position[0]
                    self.available_subwin_dict['csv_widget'] = WMML_csv(path_data_position)
                    self.available_subwin_dict['csv_widget'].get_receiver.connect(self.send_potential_receiver)
                    self.available_subwin_dict['csv_widget'].receiver_and_data_signal.connect(
                        self.send_receiver_and_data)
                    self.receiver_signal_csv_widget.connect(self.available_subwin_dict['csv_widget'].get_receivers)
                    self.subw_ma.addSubWindow(self.available_subwin_dict['csv_widget'])
                    self.available_subwin_dict['csv_widget'].show()

    @QtCore.pyqtSlot()
    def on_fe_prep_btn_clicked(self):
        try:
            if self.available_subwin_dict['prep_widget'].isVisible():
                self.available_subwin_dict['prep_widget'].show()
            else:
                self.available_subwin_dict['prep_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['prep_widget'] = WMML_prep()
            self.available_subwin_dict['prep_widget'].get_receiver.connect(self.send_potential_receiver)
            self.available_subwin_dict['prep_widget'].receiver_and_data_signal.connect(self.send_receiver_and_data)
            self.receiver_signal_prep_widget.connect(self.available_subwin_dict['prep_widget'].get_receivers)
            self.subw_ma.addSubWindow(self.available_subwin_dict['prep_widget'])
            self.available_subwin_dict['prep_widget'].show()
        pass

    @QtCore.pyqtSlot()
    def on_fe_corr_btn_clicked(self):
        try:
            if self.available_subwin_dict['corr_widget'].isVisible():
                self.available_subwin_dict['corr_widget'].show()
            else:
                self.available_subwin_dict['corr_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['corr_widget'] = WMML_correlation()
            self.available_subwin_dict['corr_widget'].get_receiver.connect(self.send_potential_receiver)
            self.available_subwin_dict['corr_widget'].receiver_and_data_signal.connect(self.send_receiver_and_data)
            self.subw_ma.addSubWindow(self.available_subwin_dict['corr_widget'])
            self.available_subwin_dict['corr_widget'].show()
        pass

    @QtCore.pyqtSlot()
    def on_ml_u_pca_btn_clicked(self):
        try:
            if self.available_subwin_dict['pca_widget'].isVisible():
                self.available_subwin_dict['pca_widget'].show()
            else:
                self.available_subwin_dict['pca_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['pca_widget'] = WMML_pca()
            self.available_subwin_dict['pca_widget'].get_receiver.connect(self.send_potential_receiver)
            self.available_subwin_dict['pca_widget'].receiver_and_data_signal.connect(self.send_receiver_and_data)
            self.receiver_signal_pca_widget.connect(self.available_subwin_dict['pca_widget'].get_receivers)
            self.subw_ma.addSubWindow(self.available_subwin_dict['pca_widget'])
            self.available_subwin_dict['pca_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_stk_btn_clicked(self):
        try:
            if self.available_mdwin_dict['stk_widget'].isVisible():
                self.available_mdwin_dict['stk_widget'].show()
            else:
                self.available_mdwin_dict['stk_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['stk_widget'] = WMML_stk()
            self.available_mdwin_dict['stk_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['stk_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['stk_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_rf_btn_clicked(self):
        try:
            if self.available_mdwin_dict['rf_widget'].isVisible():
                self.available_mdwin_dict['rf_widget'].show()
            else:
                self.available_mdwin_dict['rf_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['rf_widget'] = WMML_rf()
            self.available_mdwin_dict['rf_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['rf_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['rf_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_knn_btn_clicked(self):
        try:
            if self.available_mdwin_dict['knn_widget'].isVisible():
                self.available_mdwin_dict['knn_widget'].show()
            else:
                self.available_mdwin_dict['knn_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['knn_widget'] = WMML_knn()
            self.available_mdwin_dict['knn_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['knn_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['knn_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_gbdt_btn_clicked(self):
        try:
            if self.available_mdwin_dict['gbdt_widget'].isVisible():
                self.available_mdwin_dict['gbdt_widget'].show()
            else:
                self.available_mdwin_dict['gbdt_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['gbdt_widget'] = WMML_gbdt()
            self.available_mdwin_dict['gbdt_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['gbdt_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['gbdt_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_mlp_btn_clicked(self):
        try:
            if self.available_mdwin_dict['nn_widget'].isVisible():
                self.available_mdwin_dict['nn_widget'].show()
            else:
                self.available_mdwin_dict['nn_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['nn_widget'] = WMML_nn()
            self.available_mdwin_dict['nn_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['nn_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['nn_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_r_lr_btn_clicked(self):
        try:
            if self.available_mdwin_dict['sr_widget'].isVisible():
                self.available_mdwin_dict['sr_widget'].show()
            else:
                self.available_mdwin_dict['sr_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_mdwin_dict['sr_widget'] = WMML_sr()
            self.available_mdwin_dict['sr_widget'].model_signal.connect(self.send_model_to_train_and_test)
            self.subw_ma.addSubWindow(self.available_mdwin_dict['sr_widget'],
                                      Qt.WindowFlags(Qt.MSWindowsFixedSizeDialogHint))
            self.available_mdwin_dict['sr_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_el_btn_clicked(self):
        try:
            if self.available_subwin_dict['train_and_test_widget'].isVisible():
                self.available_subwin_dict['train_and_test_widget'].show()
            else:
                self.available_subwin_dict['train_and_test_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['train_and_test_widget'] = WMML_tas()
            self.available_subwin_dict['train_and_test_widget'].model_signal.connect(self.send_model_to_prediction)
            self.subw_ma.addSubWindow(self.available_subwin_dict['train_and_test_widget'])
            self.available_subwin_dict['train_and_test_widget'].show()

    @QtCore.pyqtSlot()
    def on_ml_op_bg_btn_clicked(self):
        try:
            if self.available_subwin_dict['train_and_test_widget'].isVisible():
                self.available_subwin_dict['train_and_test_widget'].show()
                self.available_subwin_dict['train_and_test_widget'].tabWidget.setCurrentIndex(2)
            else:
                self.available_subwin_dict['train_and_test_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['train_and_test_widget'] = WMML_tas()
            self.available_subwin_dict['train_and_test_widget'].tabWidget.setCurrentIndex(2)
            self.available_subwin_dict['train_and_test_widget'].model_signal.connect(self.send_model_to_prediction)
            self.subw_ma.addSubWindow(self.available_subwin_dict['train_and_test_widget'])
            self.available_subwin_dict['train_and_test_widget'].show()

    @QtCore.pyqtSlot()
    def on_tr_pred_btn_clicked(self):
        try:
            if self.available_subwin_dict['prediction_widget'].isVisible():
                self.available_subwin_dict['prediction_widget'].show()
            else:
                self.available_subwin_dict['prediction_widget'].showMaximized()
        except Exception as e:
            self.e = e
            self.available_subwin_dict['prediction_widget'] = WMML_pred()
            self.subw_ma.addSubWindow(self.available_subwin_dict['prediction_widget'])
            self.available_subwin_dict['prediction_widget'].show()

    @QtCore.pyqtSlot()
    def on_tr_ldmd_btn_clicked(self):
        try:
            if self.available_subwin_dict['prediction_widget'].isVisible():
                self.available_subwin_dict['prediction_widget'].show()
            else:
                self.available_subwin_dict['prediction_widget'].showMaximized()
        except Exception as e:
            self.e = e
            file_diag = QtWidgets.QFileDialog()
            md_position = QtWidgets.QFileDialog.getOpenFileName(file_diag, 'select file', '', 'pkcls files(*.pkcls)',
                                                                options=options)
            if isinstance(md_position, tuple):
                if md_position[0] == '':
                    pass
                else:
                    path_data_position = md_position[0]
                    self.available_subwin_dict['prediction_widget'] = WMML_pred(path_data_position)
                    self.subw_ma.addSubWindow(self.available_subwin_dict['prediction_widget'])
                    self.available_subwin_dict['prediction_widget'].show()

    def send_receiver_and_data(self, receiver, features, metas=None, targets=None):
        self.available_subwin_dict[receiver].get_data_from_sender(features, metas, targets)

    def send_model_to_train_and_test(self, model, receiver='train_and_test_widget'):
        if self.available_subwin_dict[receiver] is None:
            pass
        else:
            self.available_subwin_dict[receiver].get_model_from_sender(model)

    def send_model_to_prediction(self, model, receiver='prediction_widget'):
        if self.available_subwin_dict[receiver] is None:
            pass
        else:
            self.available_subwin_dict[receiver].get_model_from_sender(model)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = WMML_Main()
    main_window.show()
    sys.exit(app.exec_())
