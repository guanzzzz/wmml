from subwin.wmmlprep import Ui_WMML_prep
from subwin.wmmlsubeng import table_view, get_selection, WMML_signal
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pandas as pd
import numpy as np
from Orange.data import Table
from Orange.widgets.data.owpreprocess import Scale
import warnings
warnings.filterwarnings('ignore')


class WMML_prep(QWidget, Ui_WMML_prep):
    """
    WMML Dataset Scaler Widget
    Input:
    features: training features, type: array_like object
    metas: training metas, type: array_like object
    targets: training targets, type: array_like object
    Output signal: get_receiver_signal
    signal source: 'prep_widget' string
    Output signal: receiver_and_data_signal
    receivers: available receiver list
    features: scaled training features, type: array_like object
    metas: training metas, type: array_like object
    targets: training targets, type: array_like object
    """
    receiver_and_data_signal = pyqtSignal(str, object, object, object)
    get_receiver = pyqtSignal(str)

    def __init__(self):
        super(WMML_prep, self).__init__()
        self.setupUi(self)
        self.receiver_list = None
        self.set_reveiver_widget = None
        self.fd_model = None
        self.td_model = None
        self.d = pd.DataFrame()
        self.jz_targets = pd.DataFrame()
        self.jz_metas = pd.DataFrame()
        self.jz_fd_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_fd_tab.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.jz_td_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_td_tab.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.jzfucbx.setDisabled(True)
        self.jzadfb.setDisabled(True)
        self.jzadfun.setDisabled(True)
        self.jzfuncedit.setDisabled(True)

    @pyqtSlot()
    def on_jzadfb_clicked(self):
        cols = get_selection(self.jz_fd_tab)
        if len(cols) == 0:
            self.infobox.setText("No Feature Selected！")
        else:
            col_to_add_list = [self.d.columns.values.tolist()[i] for i in cols]
            col_to_add = col_to_add_list[0]
            self.jzfuncedit.insertPlainText("self.d[{}]".format(col_to_add))

    def get_data_from_sender(self, features, metas=None, targets=None):
        if metas is not None:
            self.jz_targets = targets
            self.jz_metas = metas
        self.d = features
        self.fd_model = table_view(self.jz_fd_tab, self.d)
        self.td_model = table_view(self.jz_td_tab, self.jz_targets)

    def get_receivers(self, receiver_list):
        self.receiver_list = receiver_list
        self.set_reveiver_widget = WMML_signal(self.receiver_list)
        self.set_reveiver_widget.send_receiver_signal.connect(self.send_receiver_selection)
        self.set_reveiver_widget.show()

    @pyqtSlot()
    def on_jzsdf_clicked(self):
        self.get_receiver.emit('prep_widget')

    def send_receiver_selection(self, selection):
        self.receiver_and_data_signal.emit(selection, self.d, self.jz_metas, self.jz_targets)
        self.infobox.setText('Data has been successfully transferred')

    @pyqtSlot()
    def on_jzadfun_clicked(self):
        func_attr = self.jzfucbx.currentText()
        if func_attr == 'Ln':
            self.jzfuncedit.insertPlainText('np.log()')
        elif func_attr == 'Log10':
            self.jzfuncedit.insertPlainText('np.log10()')
        elif func_attr == 'Log2':
            self.jzfuncedit.insertPlainText('np.log2()')
        elif func_attr == 'Sin':
            self.jzfuncedit.insertPlainText('np.sin()')
        elif func_attr == 'Cos':
            self.jzfuncedit.insertPlainText('np.cos()')
        elif func_attr == 'Tan':
            self.jzfuncedit.insertPlainText('np.tan()')
        elif func_attr == 'Exp':
            self.jzfuncedit.insertPlainText('np.exp()')
        elif func_attr == 'Arctan':
            self.jzfuncedit.insertPlainText('np.arctan()')
        elif func_attr == 'Arcsin':
            self.jzfuncedit.insertPlainText('np.arcsin()')
        elif func_attr == 'Arccos':
            self.jzfuncedit.insertPlainText('np.arccos()')

    @pyqtSlot()
    def on_jzcptfb_clicked(self):
        data = Table(self.d)
        method = {"method": None}
        if self.jzns.isChecked():
            if self.jzns_std.isChecked():
                method["method"] = 1  # "Standardize to μ=0, σ²=1"
                scale = Scale.createinstance(method)
                scaled_table = scale(data)
            elif self.jzns_ctr.isChecked():
                method["method"] = 0  # "Center to μ=0"
                scale = Scale.createinstance(method)
                scaled_table = scale(data)
            elif self.jzns_sle.isChecked():
                method["method"] = 2  # "Scale to σ²=1"
                scale = Scale.createinstance(method)
                scaled_table = scale(data)
            elif self.jzns_nmlf.isChecked():
                method["method"] = 4  # "Normalize to interval [-1, 1]"
                scale = Scale.createinstance(method)
                scaled_table = scale(data)
            else:  # self.jzns_nmlz.isChecked():
                method["method"] = 3  # "Normalize to interval [0, 1]"
                scale = Scale.createinstance(method)
                scaled_table = scale(data)
            d = pd.DataFrame(scaled_table.X, columns=self.d.columns.values.tolist())
            self.d = d
            self.fd_model = table_view(self.jz_fd_tab, self.d)
        else:
            args = self.jzfuncedit.toPlainText()
            if args == '':
                pass
            else:
                try:
                    new_feature_column = eval(args)
                    if self.nfnld.text() == '':
                        new_feature_name = 'new_feature'
                    else:
                        new_feature_name = self.nfnld.text()
                    self.d = pd.concat([self.d, pd.DataFrame(new_feature_column, columns=[new_feature_name])], axis=1)
                except Exception as e:
                    self.infobox.setText(str(e))
                    # pprint(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_prep()
    x, y = np.linspace(1, 10, 10), np.linspace(1, 100, 10)
    X, Y = pd.DataFrame(x, columns=['X']), pd.DataFrame(y, columns=['Y'])
    win.get_data_from_sender(X, pd.DataFrame(), Y)
    win.show()
    sys.exit(app.exec_())
