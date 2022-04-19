from subwin.wmmlsubeng import *
from subwin.wmmlsr import Ui_WMML_lr
from sklearn import linear_model, metrics
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


class PRThread(QThread):
    """
    WMML polynomial regression thread
    available Input:
    data: X, Y, type: array-like object
    model: selected linear regression model
    Meta_calc_signal:
    meta_calculation signal
    float: real-time R-square
    str: real-time calculation rows
    Output_signal:
    transformed X: numerical matrix
    transformed X: numerical matrix
    feature coefficient: a list of feature coefficient for each fitted model
    indices: R-square for each fitted model
    """
    Output_signal = pyqtSignal(object, object, object, object)
    Meta_calc_signal = pyqtSignal(float, int)

    def __init__(self, features, targets, model):
        super().__init__()
        self.x = features
        self.y = targets
        self.ml_model = model
        self.e = None

    def run(self):
        op_X = {}
        op_Y = {}
        coef_xy = {}
        r2_xy = []
        for i in range(self.x.shape[0]):
            org_x = {}
            for j in range(self.ml_model.parameters['order'] + 1):
                org_x['order{}'.format(j)] = pow(self.x.iloc[[i]].dropna(axis=1).values.ravel(), j)
            temp_x = pd.DataFrame.from_dict(org_x).values
            temp_y = self.y.iloc[[i]].dropna(axis=1).values.ravel()
            temp_model = self.ml_model.model
            temp_model.fit(temp_x, temp_y)
            op_X[i] = org_x['order1'].ravel()
            op_Y[i] = temp_model.predict(temp_x)
            coef_xy[i] = temp_model.coef_
            r2_xy.append(metrics.r2_score(temp_y, op_Y[i]))
            if self.ml_model.parameters['nums'] is None:
                pass
            else:
                org_xp = {}
                if self.ml_model.parameters['opts'] == 'Uniform Linear':
                    org_xps = np.linspace(start=org_x['order1'].min(), stop=org_x['order0'].max(),
                                          num=self.ml_model.parameters['nums'])
                elif self.ml_model.parameters['opts'] == 'Log2':
                    org_xps = np.logspace(start=org_x['order1'].min(), stop=org_x['order0'].max(),
                                          num=self.ml_model.parameters['nums'], base=2)
                elif self.ml_model.parameters['opts'] == 'Log10':
                    org_xps = np.logspace(start=org_x['order1'].min(), stop=org_x['order0'].max(),
                                          num=self.ml_model.parameters['nums'], base=10)
                else:
                    org_xps = np.logspace(start=org_x['order1'].min(), stop=org_x['order0'].max(),
                                          num=self.ml_model.parameters['nums'], base=np.e)
                for j in range(self.ml_model.parameters['order'] + 1):
                    org_xp['order{}'.format(j)] = pow(org_xps, j)
                temp_xp = pd.DataFrame.from_dict(org_xp).values
                op_X[i] = org_xps
                op_Y[i] = temp_model.predict(temp_xp)

            self.Meta_calc_signal.emit(r2_xy[i], i)
        self.Output_signal.emit(op_X, op_Y, coef_xy, r2_xy)


class WMML_sr(QWidget, Ui_WMML_lr):
    """
    WMML polynomial regression widget
    available Input:
    data: X, Y, type: array-like object
    Output:
    transformed X: numerical matrix
    model: Model object
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_sr, self).__init__()
        self.setupUi(self)
        self.model = Model()
        self.ip_x = self.ip_y = self.op_X = self.op_Y = self.ip_x_md = self.op_md = None
        self.coef_xy = self.coef_xy_md = self.r2_xy = self.r2_xy_md = self.ip_y_md = None
        self.prtrd = None
        self.jzsdf.clicked.connect(self.get_selection_state)
        self.groupBox_8.setDisabled(True)
        self.groupBox_5.setDisabled(True)
        self.groupBox_9.setDisabled(True)
        self.l1_ratio_validator = QDoubleValidator()
        self.l1_ratio_validator.setRange(0.00, 1.00, 2)
        self.l1_ratio_validator.setNotation(QDoubleValidator.StandardNotation)
        self.lasso_alpha.setValidator(QDoubleValidator())
        self.rr_cv.setValidator(QIntValidator())
        self.rr_alpha.setValidator(QDoubleValidator())
        self.enr_alpha.setValidator(QDoubleValidator())
        self.enr_l1.setValidator(self.l1_ratio_validator)
        self.tp_opn.setValidator(QIntValidator())
        self.iptx.setEditTriggers(QTableView.NoEditTriggers)
        self.iptx.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ipty.setEditTriggers(QTableView.NoEditTriggers)
        self.ipty.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.optx.setEditTriggers(QTableView.NoEditTriggers)
        self.optx.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.opty.setEditTriggers(QTableView.NoEditTriggers)
        self.opty.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.opids.setEditTriggers(QTableView.NoEditTriggers)
        self.opids.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.opcofxy.setEditTriggers(QTableView.NoEditTriggers)
        self.opcofxy.setSelectionBehavior(QAbstractItemView.SelectRows)

    @pyqtSlot(int)
    def on_tp_odr_valueChanged(self, intx):
        self.tp_fo.setText(str(intx))

    @pyqtSlot()
    def on_jzcpt_clicked(self):
        msb = QMessageBox()
        reply = QMessageBox.question(msb, 'Message', 'The model and data will be rewrite Proceed?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.define_model('yes', False)
        else:
            pass

    def get_selection_state(self):
        msb = QMessageBox()
        reply = QMessageBox.question(msb, 'Message', 'Model will be rewrite! Proceed?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.define_model('yes', True)
        else:
            pass

    def define_model(self, selection, is_send):
        try:
            if selection == "yes":
                self.model.model_classes = "r"
                self.model.opt_selection = False
                if self.sm_lr.isChecked():
                    self.model.model_names = "Linear Regression"
                    self.model.model_names_eng = "lr"
                    if self.lr_fit.isChecked():
                        self.model.parameters['fit_intercept'] = True
                    else:
                        self.model.parameters['fit_intercept'] = False
                elif self.sm_lasso.isChecked():
                    self.model.model_names = "LASSO"
                    self.model.model_names_eng = "LASSO"
                    if self.lasso_fit.isChecked():
                        self.model.parameters['fit_intercept'] = True
                    else:
                        self.model.parameters['fit_intercept'] = False
                    try:
                        self.model.parameters['alpha'] = eval(self.lasso_alpha.text())
                    except Exception as e:
                        print(str(e))
                        self.lasso_alpha.setText("1.0")
                        self.model.parameters['alpha'] = eval(self.lasso_alpha.text())
                elif self.sm_rr.isChecked():
                    self.model.model_names = "Ridge Regression"
                    self.model.model_names_eng = "rr"
                    if self.rr_fit.isChecked():
                        self.model.parameters['fit_intercept'] = True
                    else:
                        self.model.parameters['fit_intercept'] = False
                    try:
                        self.model.parameters['alphas'] = eval(self.rr_alpha.text())
                    except Exception as e:
                        print(str(e))
                        self.rr_alpha.setText('(0.1, 1.0, 10.0)')
                        self.model.parameters['alphas'] = eval(self.rr_alpha.text())
                    try:
                        self.model.parameters['cv'] = eval(self.rr_cv.text())
                    except Exception as e:
                        print(str(e))
                        self.rr_cv.setText('None')
                        self.model.parameters['cv'] = eval(self.rr_cv.text())
                    if self.rr_gcv_auto.isChecked():
                        self.model.parameters['gcv_mode'] = 'auto'
                    elif self.rr_gcv_svd.isChecked():
                        self.model.parameters['gcv_mode'] = 'svd'
                    else:
                        self.model.parameters['gcv_mode'] = 'eigen'
                elif self.sm_enr.isChecked():
                    self.model.model_names = "Elastic Net Regression"
                    self.model.model_names_eng = "eln"
                    if self.enr_fit.isChecked():
                        self.model.parameters['fit_intercept'] = True
                    else:
                        self.model.parameters['fit_intercept'] = False
                    try:
                        self.model.parameters['alpha'] = eval(self.enr_alpha.text())
                    except Exception as e:
                        print(str(e))
                        self.enr_alpha.setText('1.0')
                        self.model.parameters['alpha'] = eval(self.enr_alpha.text())
                    try:
                        self.model.parameters['l1_ratio'] = eval(self.enr_l1.text())
                    except Exception as e:
                        print(str(e))
                        self.enr_l1.setText('0.5')
                        self.model.parameters['l1_ratio'] = eval(self.enr_l1.text())
                if self.model.model_names == "Linear Regression":
                    self.model.model = linear_model.LinearRegression(
                        fit_intercept=self.model.parameters['fit_intercept']
                    )
                elif self.model.model_names == "LASSO":
                    self.model.model = linear_model.Lasso(
                        alpha=self.model.parameters['alpha'],
                        fit_intercept=self.model.parameters['fit_intercept']
                    )
                elif self.model.model_names == 'Ridge Regression':
                    self.model.model = linear_model.RidgeCV(
                        alphas=self.model.parameters['alphas'],
                        fit_intercept=self.model.parameters['fit_intercept'],
                        cv=self.model.parameters['cv'],
                        gcv_mode=self.model.parameters['gcv_mode']
                    )
                elif self.model.model_names == 'Elastic Net Regression':
                    self.model.model = linear_model.ElasticNet(
                        alpha=self.model.parameters['alpha'],
                        l1_ratio=self.model.parameters['l1_ratio'],
                        fit_intercept=self.model.parameters['fit_intercept']
                    )
                if is_send:
                    self.model_signal[object].emit(self.model)
                else:
                    if self.ip_x.empty or self.ip_y.empty:
                        self.infobox.setText('No active features or targets detected')
                    else:
                        self.model.parameters['order'] = int(self.tp_fo.text())
                        if self.tp_opn.text() == 'None':
                            self.model.parameters['nums'] = None
                        else:
                            self.model.parameters['nums'] = abs(int(self.tp_opn.text()))
                        self.model.parameters['opts'] = self.tp_opt.currentText()
                        self.rmsev.clear()
                        self.prtrd = PRThread(self.ip_x, self.ip_y, self.model)
                        self.prtrd.Output_signal.connect(self.compr_complete)
                        self.prtrd.Meta_calc_signal.connect(self.show_real_time_calc)
                        self.prtrd.start()
        except Exception as e:
            self.infobox.setText("Error :" + str(e))

    def show_real_time_calc(self, r2s, nuc):
        self.rmsev.append('No{numb} rows R-square:{r2ss}'.format(numb=nuc, r2ss=r2s))

    def compr_complete(self, t_x, t_y, coef_xy, r2_xy):
        self.op_X = pd.DataFrame()
        self.op_Y = pd.DataFrame()
        for key in t_x.keys():
            t = pd.DataFrame(t_x[key])
            self.op_X = pd.concat([self.op_X, t], axis=1)
        for key in t_y.keys():
            t = pd.DataFrame(t_y[key])
            self.op_Y = pd.concat([self.op_Y, t], axis=1)
        self.coef_xy = pd.DataFrame.from_dict(coef_xy)
        self.op_X = self.op_X.T
        self.op_Y = self.op_Y.T
        self.coef_xy = self.coef_xy.T
        self.rmsev.append('Overall R-square:{r2ss}'.format(r2ss=np.array(r2_xy).mean()))
        self.r2_xy = pd.DataFrame(r2_xy, columns=['R-square'])
        self.op_md = table_view(self.optx, self.op_X)
        self.op_md = table_view(self.opty, self.op_Y)
        self.coef_xy_md = table_view(self.opcofxy, self.coef_xy)
        self.r2_xy_md = table_view(self.opids, self.r2_xy)
        self.infobox.setText("Compute Complete")

    @pyqtSlot()
    def on_od_sx_clicked(self):
        if self.op_X is None:
            self.infobox.setText("No Active Data Detected")
        else:
            file_diag = QFileDialog()
            save_position = QFileDialog.getSaveFileName(file_diag, 'save file', './', 'csv files(*.csv)',
                                                        options=options)
            if isinstance(save_position, tuple):
                path_save_position = save_position[0]
                if path_save_position == '':
                    pass
                else:
                    self.op_X.to_csv(path_save_position + '.csv')
            else:
                pass

    @pyqtSlot()
    def on_od_syy_clicked(self):
        if self.op_Y is None:
            self.infobox.setText("No Active Data Detected")
        else:
            file_diag = QFileDialog()
            save_position = QFileDialog.getSaveFileName(file_diag, 'save file', './', 'csv files(*.csv)',
                                                        options=options)
            if isinstance(save_position, tuple):
                path_save_position = save_position[0]
                if path_save_position == '':
                    pass
                else:
                    self.op_Y.to_csv(path_save_position + '.csv')
            else:
                pass

    @pyqtSlot()
    def on_od_sy_clicked(self):
        if self.opty is None:
            self.infobox.setText("No Active Data Detected")

    @pyqtSlot()
    def on_id_lx_clicked(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'csv files( *.csv)',
                                                        options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                if path_data_position == '':
                    pass
                else:
                    self.ip_x = pd.read_csv(path_data_position)
                    self.ip_x_md = table_view(self.iptx, self.ip_x)
                    self.infobox.setText(path_data_position)
        except Exception as e:
            print(e)

    @pyqtSlot()
    def on_id_ly_clicked(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'csv files( *.csv)',
                                                        options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                if path_data_position == '':
                    pass
                else:
                    self.ip_y = pd.read_csv(path_data_position)
                    self.ip_y_md = table_view(self.ipty, self.ip_y)
                    self.infobox.setText(path_data_position)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_sr()
    win.show()
    sys.exit(app.exec_())
