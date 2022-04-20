from subwin.wmmlcsv import Ui_WMML_csv
from subwin.wmmlsubeng import table_view, get_selection, WMML_signal, options
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pandas as pd
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')


class WMML_csv(QWidget, Ui_WMML_csv):
    receiver_and_data_signal = pyqtSignal(str, object, object, object)
    get_receiver = pyqtSignal(str)

    def __init__(self, csv_file):
        super(WMML_csv, self).__init__()
        self.setupUi(self)
        self.receiver_list = None
        self.set_reveiver_widget = None
        self.csv = pd.read_csv(csv_file)
        self.infobox.setText(csv_file)
        self.od_model = table_view(self.jz_od_tab, self.csv)
        self.fd_model = None
        self.td_model = None
        self.jz_features = pd.DataFrame()
        self.jz_targets = pd.DataFrame()
        self.jz_metas = pd.DataFrame()
        self.jz_od_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_od_tab.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.jz_fd_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_fd_tab.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.jz_td_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_td_tab.setSelectionBehavior(QAbstractItemView.SelectColumns)

    def reloadui(self):
        try:
            file_diag = QFileDialog()
            csv_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'pkcls files(*.pkcls)',
                                                       options=options)
            if isinstance(csv_position, tuple):
                csv_file = csv_position[0]
                print("csv_file:{}".format(csv_file))
                if csv_file == '':
                    pass
                else:
                    self.csv = pd.read_csv(csv_file)
                    self.infobox.setText(csv_file)
                    self.od_model = table_view(self.jz_od_tab, self.csv)
            else:
                pass
        except Exception as e:
            self.infobox.setText(str(e))

    @pyqtSlot()
    def on_jzrld_btn_clicked(self):
        self.reloadui()

    @pyqtSlot()
    def on_jzatm_btn_clicked(self):
        try:
            all_features = self.csv.columns.values.tolist()
            if 'target' in all_features:
                if 'class' in all_features:
                    all_features.remove('class')
                    self.jz_metas = self.csv[['class']]
                all_features.remove('target')
                self.jz_features = self.csv[all_features]
                self.jz_targets = self.csv[['target']]
                self.fd_model = table_view(self.jz_fd_tab, self.jz_features)
                self.td_model = table_view(self.jz_td_tab, self.jz_targets)
                self.infobox.setText("Auto Match Complete！")
            else:
                self.infobox.setText("Auto Match Failed！Please name the target column as 'target'")
        except Exception as e:
            self.infobox.setText(str(e))

    @pyqtSlot()
    def on_jzadfb_clicked(self):
        try:
            cols = get_selection(self.jz_od_tab)
            if len(cols) == 0:
                self.infobox.setText("No Feature Selected！")
            else:
                col_to_add_list = [self.csv.columns.values.tolist()[i] for i in cols]
                features_to_add_df = self.csv[col_to_add_list]
                self.jz_features = pd.concat([self.jz_features, features_to_add_df], axis=1)
                self.fd_model = table_view(self.jz_fd_tab, self.jz_features)
                self.infobox.setText("Feature Added！")
        except Exception as e:
            self.infobox.setText(str(e))

    @pyqtSlot()
    def on_jzrmfb_clicked(self):
        try:
            cols = get_selection(self.jz_fd_tab)
            if len(cols) == 0:
                self.infobox.setText("No Feature Selected！")
            else:
                col_to_remove_list = [self.jz_features.columns.values.tolist()[i] for i in cols]
                features_to_remove_df = self.jz_features[col_to_remove_list]
                self.jz_features = self.jz_features.drop(features_to_remove_df, axis=1)
                self.fd_model = table_view(self.jz_fd_tab, self.jz_features)
                self.infobox.setText("Feature Removed！")
        except Exception as e:
            pprint(str(e))
            self.infobox.setText("No Feature Selected！")

    @pyqtSlot()
    def on_jzrmtgt_clicked(self):
        try:
            self.jz_targets = pd.DataFrame()
            self.td_model = table_view(self.jz_td_tab, self.jz_targets)
            self.infobox.setText("Targets Removed！")
        except Exception as e:
            self.infobox.setText(str(e))

    @pyqtSlot()
    def on_jzstb_clicked(self):
        try:
            cols = get_selection(self.jz_od_tab)
            if len(cols) > 1:
                self.infobox.setText("There can only be 1 target variable！")
            elif len(cols) == 0:
                self.infobox.setText("No Feature Selected！")
            else:
                col_to_add_list = [self.csv.columns.values.tolist()[i] for i in cols]
                self.jz_targets = self.csv[col_to_add_list]
                self.td_model = table_view(self.jz_td_tab, self.jz_targets)
                self.infobox.setText("Target Selected！")
        except Exception as e:
            self.infobox.setText(str(e))

    @pyqtSlot()
    def on_jzsdf_clicked(self):
        self.get_receiver.emit('csv_widget')
        # pprint(self.get_receiver)

    def get_receivers(self, receiver_list):
        self.receiver_list = receiver_list
        self.set_reveiver_widget = WMML_signal(self.receiver_list)
        self.set_reveiver_widget.send_receiver_signal.connect(self.send_receiver_selection)
        self.set_reveiver_widget.show()

    def send_receiver_selection(self, selection):
        if selection == '':
            pass
        else:
            self.receiver_and_data_signal.emit(selection, self.jz_features, self.jz_metas, self.jz_targets)
            self.infobox.setText('Data has been successfully transferred')

    def get_data_from_sender(self, features, metas=None, targets=None):
        if metas is not None:
            self.jz_targets = targets
            self.jz_metas = metas
        self.jz_features = features
        self.fd_model = table_view(self.jz_fd_tab, self.jz_features)
        self.td_model = table_view(self.jz_td_tab, self.jz_targets)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_csv('data.csv')
    win.show()
    sys.exit(app.exec_())
