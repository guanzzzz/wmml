from subwin.wmmlpca import Ui_WMML_pca
from subwin.wmmlsubeng import table_view, WMML_signal
from Orange.data import Table, Domain
from Orange.widgets.widget import Input
from Orange.widgets.unsupervised.owpca import OWPCA
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class WMML_OW_PCA(OWPCA):
    """
    The PCA widget wrapped Orange PCA widget
    Input:
    features: training features, type: array_like object
    metas: training metas, type: array_like object
    targets: training targets, type: array_like object
    Output signal: transformed_data
    features:  transformed feature data
    """
    transformed_data = pyqtSignal(object)

    def commit(self):
        transformed = None
        if self._pca is not None:
            if self._transformed is None:
                self._transformed = self._pca(self.data)
            transformed = self._transformed
            domain = Domain(
                transformed.domain.attributes[:self.ncomponents],
                self.data.domain.class_vars,
                self.data.domain.metas
            )
            transformed = transformed.from_table(domain, transformed)
        self.transformed_data.emit(pd.DataFrame(transformed, columns=['PC{}'.format(i + 1)
                                                                      for i in range(self.ncomponents)]))


class WMML_pca(QWidget, Ui_WMML_pca):
    """
    WMML PCA widget
    Input:
    features: training features, type: array_like object
    metas: training metas, type: array_like object
    targets: training targets, type: array_like object
    Output signal: get_receiver_signal
    signal source: 'pca_widget' string
    Output signal: receiver_and_data_signal
    receivers: available receiver list
    features: scaled training features, type: array_like object
    metas: training metas, type: array_like object
    targets: training targets, type: array_like object
    """
    receiver_and_data_signal = pyqtSignal(str, object, object, object)
    get_receiver = pyqtSignal(str)

    class Inputs:
        data = Input("Data", Table)

    def __init__(self):
        super(WMML_pca, self).__init__()
        self.setupUi(self)
        self.receiver_list = None
        self.set_reveiver_widget = None
        self.jz_features = pd.DataFrame()
        self.jz_targets = pd.DataFrame()
        self.jz_metas = pd.DataFrame()
        self.fd_model = None
        self.pca_transformed = pd.DataFrame()
        self.pca_widget = WMML_OW_PCA()
        self.pca_widget.transformed_data.connect(self.set_data)
        self.pca_container.addWidget(self.pca_widget)
        self.pca_transformed_model = None
        self.data_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.data_tab_2.setEditTriggers(QTableView.NoEditTriggers)

    @Inputs.data
    def set_data(self, data):
        self.pca_transformed = data
        self.pca_transformed_model = table_view(self.data_tab_2, data)

    @pyqtSlot()
    def on_jzsdf_clicked(self):
        self.get_receiver.emit('pca_widget')
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
            if self.pca_transformed.empty:
                pass
            else:
                self.receiver_and_data_signal.emit(selection, self.pca_transformed, None, None)
                self.infobox.setText('Data has been successfully transferred')

    def get_data_from_sender(self, features, metas, targets):
        self.jz_features = features
        self.jz_targets = targets
        self.jz_metas = metas
        self.fd_model = table_view(self.data_tab, self.jz_features)
        self.pca_widget.set_data(Table(features))
        self.infobox.setText('Data has been successfully transferred')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_pca()
    x, y = np.linspace(1, 10, 10), np.linspace(1, 100, 10)
    X, Y = pd.DataFrame(x, columns=['X']), pd.DataFrame(y, columns=['Y'])
    win.get_data_from_sender(pd.concat([X, Y], axis=1), pd.DataFrame(), pd.DataFrame())
    win.show()
    sys.exit(app.exec_())
