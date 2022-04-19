from subwin.wmmlcorrelation import Ui_WMML_correlation
from subwin.wmmlsubeng import table_view, HTM, TXT, options
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn import ensemble, linear_model
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from minepy import MINE
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class CorreThread(QThread):
    """
    Correlation Analysis thread
    Input:
    features: training features, type: array_like object
    targets: training targets, type: array_like object
    correlation_index: the type of correlation index to be calculated
    Output signal:
    correlation index: calculated correlation indices
    type of correlation index: the type of correlation index calculated
    """
    Output_signal = pyqtSignal(object, object)

    def __init__(self, features, targets, correlation_index):
        super().__init__()
        self.fe_features = features
        self.fe_targets = targets
        self.fe_original_dataset = pd.concat([features, targets], axis=1)
        self.correlation_index = correlation_index
        self.fe_corr_fig = None

    def run(self):
        try:
            if self.correlation_index == 'Pearson correlation coefficient':
                temp_dataset = self.fe_original_dataset.astype('float64')
                fe_corr_index = temp_dataset.corr(method='pearson')
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == 'Pearson correlation coefficient-square root':
                temp_dataset = self.fe_original_dataset.astype('float64')
                temp_dataset = temp_dataset.corr(method='pearson')
                fe_corr_index = temp_dataset.pow(2).pow(0.5)
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == "'Kendall's tau coefficient":
                temp_dataset = self.fe_original_dataset.astype('float64')
                fe_corr_index = temp_dataset.corr(method='kendall')
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == 'Spearman rank correlation':
                temp_dataset = self.fe_original_dataset.astype('float64')
                fe_corr_index = temp_dataset.corr(method='spearman')
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == 'Feature importance-by random forest':
                clf = ensemble.RandomForestRegressor()
                clf.fit(self.fe_features, self.fe_targets.values.ravel())
                feature_importance_array = clf.feature_importances_
                temp_list = self.fe_features.columns.values.tolist()
                fe_corr_index = pd.concat([pd.DataFrame(data=feature_importance_array, columns=['importance']),
                                          pd.DataFrame(data=temp_list, columns=['features'])], axis=1)
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == 'Feature importance-by extra tree':
                clf = ensemble.ExtraTreesRegressor()
                clf.fit(self.fe_features, self.fe_targets.values.ravel())
                feature_importance_array = clf.feature_importances_
                fe_corr_index = pd.concat([pd.DataFrame(data=feature_importance_array, columns=['importance']),
                                          pd.DataFrame(data=self.fe_features.columns.values.tolist(),
                                                       columns=['features'])], axis=1)
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            elif self.correlation_index == 'Linear regression coefficient':
                clf = linear_model.LinearRegression()
                clf.fit(self.fe_features, self.fe_targets.values.ravel())
                fe_corr_index = pd.concat([pd.DataFrame(data=clf.coef_, columns=['importance']),
                                           pd.DataFrame(data=self.fe_features.columns.values.tolist(),
                                                        columns=['features'])], axis=1)
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            else:
                corr_index_dict = {}
                for columnx in self.fe_original_dataset.columns:
                    corr_indexs = []
                    for columny in self.fe_original_dataset.columns:
                        x_d, y_d = self.fe_original_dataset[columnx].values, self.fe_original_dataset[columny].values
                        mine = MINE()
                        mine.compute_score(x_d, y_d)
                        if self.correlation_index == 'Maximum asymmetry score':
                            corr_indexs.append(mine.mas())
                        elif self.correlation_index == 'Maximum edge value':
                            corr_indexs.append(mine.mev())
                        elif self.correlation_index == 'Minimum cell number':
                            corr_indexs.append(mine.mcn())
                        elif self.correlation_index == 'Generalized mean information coefficient ':
                            corr_indexs.append(mine.gmic())
                        elif self.correlation_index == 'Total information coefficient':
                            corr_indexs.append(mine.tic())
                        else:
                            corr_indexs.append(mine.mic())
                    corr_index_dict[columnx] = corr_indexs
                fe_corr_index = pd.DataFrame(corr_index_dict, index=self.fe_original_dataset.columns)
                fe_corr_index.round(4)
                self.Output_signal.emit(self.correlation_index, fe_corr_index)
            # print('fe_corr_index:{}'.format(fe_corr_index))
        except Exception as e:
            print(e)


class WMML_correlation(QWidget, Ui_WMML_correlation):
    receiver_and_data_signal = pyqtSignal(str, object, object, object)
    get_receiver = pyqtSignal(str)

    def __init__(self):
        super(WMML_correlation, self).__init__()
        self.setupUi(self)
        self.corr_index = pd.DataFrame()
        self.corr_model = None
        self.fe_corr_tab.setEditTriggers(QTableView.NoEditTriggers)
        self.jz_features = pd.DataFrame()
        self.jz_targets = pd.DataFrame()
        self.jz_metas = pd.DataFrame()
        self.correcompute = None
        self.fe_corr_figview = None
        self.fe_corr_fig = None

    def get_data_from_sender(self, features, metas, targets):
        self.jz_features = features
        self.jz_targets = targets
        self.jz_metas = metas
        self.infobox.setText('Detected Input Data')

    @pyqtSlot()
    def on_cptbtn_clicked(self):
        if self.jz_features.empty:
            pass
        else:
            self.correcompute = CorreThread(self.jz_features, self.jz_targets, self.idxsbx.currentText())
            self.correcompute.Output_signal.connect(self.get_corre_data)
            self.correcompute.run()

    def get_corre_data(self, correlation_index, corr_index):
        if correlation_index == "Feature importance-by random forest"\
                or correlation_index == 'Feature importance-by extra tree'\
                or correlation_index == 'Linear regression coefficient':
            corr_index.sort_values(by='importance', axis=0, ascending=False, inplace=True)
            self.fe_corr_figview = QGridLayout()
            self.fe_corr_figview.setObjectName("fe_corr_figview")
            self.gridLayout_4.addLayout(self.fe_corr_figview, 0, 0, 1, 1)
            self.fe_corr_fig = TXT()
            self.fe_corr_fig.plot(corr_index)
            self.fe_corr_figview.addWidget(self.fe_corr_fig)
            self.fe_corr_figview.addWidget(NavigationToolbar(self.fe_corr_fig, self))
            self.corr_index = corr_index
            self.corr_model = table_view(self.fe_corr_tab, corr_index)

        else:
            self.fe_corr_figview = QGridLayout()
            self.fe_corr_figview.setObjectName("fe_corr_figview")
            self.gridLayout_4.addLayout(self.fe_corr_figview, 0, 0, 1, 1)
            self.fe_corr_fig = HTM()
            self.fe_corr_fig.plot(corr_index)
            self.fe_corr_figview.addWidget(self.fe_corr_fig)
            self.fe_corr_figview.addWidget(NavigationToolbar(self.fe_corr_fig, self))
            self.corr_index = corr_index
            self.corr_model = table_view(self.fe_corr_tab, corr_index)

    @pyqtSlot()
    def on_fe_s_corr_btn_clicked(self):
        file_diag = QFileDialog()
        save_position = QFileDialog.getSaveFileName(file_diag, 'save file', './', 'csv files(*.csv)', options=options)
        if isinstance(save_position, tuple):
            path_save_position = save_position[0]
            if path_save_position == '':
                pass
            else:
                self.corr_index.to_csv(path_save_position)
        else:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_correlation()
    x, y = np.linspace(1, 10, 10), np.linspace(1, 100, 10)
    X, Y = pd.DataFrame(x, columns=['X']), pd.DataFrame(y, columns=['Y'])
    win.get_data_from_sender(X, pd.DataFrame(), Y)
    win.show()
    sys.exit(app.exec_())
