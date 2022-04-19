import numpy as np
import pandas as pd
from subwin.wmmlsignal import Ui_WMML_signal
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import itertools
import matplotlib
import seaborn as sns
from itertools import cycle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import warnings
warnings.filterwarnings('ignore')
matplotlib.use("Qt5Agg")
options = QFileDialog.Options()
options |= QFileDialog.DontUseNativeDialog


class WMML_signal(QWidget, Ui_WMML_signal):
    """
    WMML Signal Widget
    Available signal:
    send_receiver_signal: Receiver Widget Name
    """
    send_receiver_signal = pyqtSignal(str)

    def __init__(self, receiver_list):
        super(WMML_signal, self).__init__()
        self.setupUi(self)
        if len(receiver_list) > 0:
            self.receiver_list = receiver_list
            self.radiobuttons = {}
            for i in range(len(self.receiver_list)):
                self.radiobuttons[i] = QRadioButton()
                self.radiobuttons[i].setText(self.receiver_list[i])
                self.verticalLayout.addWidget(self.radiobuttons[i])
            self.okbtn.clicked.connect(self.get_receiver_selection)
        else:
            self.label = QLabel()
            font = QFont()
            font.setFamily("Arial")
            self.label.setFont(font)
            self.label.setText("No possible receivers detected!")
            self.verticalLayout.addWidget(self.label)

    def get_receiver_selection(self):
        selection = None
        for i in range(len(self.receiver_list)):
            if self.radiobuttons[i].isChecked():
                selection = self.receiver_list[i]
        self.send_receiver_signal.emit(selection)
        self.close()


def ranges(indices):
    g = itertools.groupby(enumerate(indices),
                          key=lambda t: t[1] - t[0])
    for _, range_ind in g:
        range_ind = list(range_ind)
        _, start = range_ind[0]
        _, end = range_ind[-1]
        yield start, end + 1


def selection_blocks(selection):
    if selection.count() > 0:
        rowranges = {range(span.top(), span.bottom() + 1)
                     for span in selection}
        colranges = {range(span.left(), span.right() + 1)
                     for span in selection}
    else:
        return [], []

    rows = sorted(set(itertools.chain(*rowranges)))
    cols = sorted(set(itertools.chain(*colranges)))
    return list(ranges(rows)), list(ranges(cols))


def get_selection(view):
    # Get selection columns of QTableView
    selmodel = view.selectionModel()
    selection = selmodel.selection()
    model = view.model()
    while isinstance(model, QAbstractProxyModel):
        selection = model.mapSelectionToSource(selection)
        model = model.sourceModel()
    _, col_spans = selection_blocks(selection)
    # rows = list(itertools.chain.from_iterable(itertools.starmap(range, row_spans)))
    cols = list(itertools.chain.from_iterable(itertools.starmap(range, col_spans)))
    return cols


def table_view(table_widget, input_table):
    # view data in QTableview
    input_table_rows = input_table.shape[0]
    input_table_columns = input_table.shape[1]
    model = QStandardItemModel(input_table_rows, input_table_columns)
    model.setHorizontalHeaderLabels([str(i) for i in input_table.columns.values.tolist()])

    for i in range(input_table_rows):
        input_table_rows_values = input_table.iloc[[i]]
        input_table_rows_values_array = np.array(input_table_rows_values)
        input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
        for j in range(input_table_columns):
            input_table_items_list = input_table_rows_values_list[j]
            input_table_items = str(input_table_items_list)
            newItem = QStandardItem(input_table_items)
            # newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            model.setItem(i, j, newItem)
    table_widget.setModel(model)
    return model


def get_current_table(table_widget, compared_table):
    compared_table_rows = compared_table.shape[0]
    compared_table_columns = compared_table.shape[1]
    for i in range(compared_table_rows):
        for j in range(compared_table_columns):
            compared_table.iloc[i, j] = table_widget.item(i, j).text()


def plot_roc_dup(lw):
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9)


class HTM(FigureCanvas):
    # Heat map for correlation analysis
    def __init__(self):
        fig = plt.figure(figsize=(5.0, 4.0), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(data, anote=False):
        plt.rc('font', family='Arial')
        xlabels = data.index.values.tolist()
        ylabels = data.columns.values.tolist()
        sns.heatmap(
            data,
            annot=anote,
            yticklabels=ylabels,
            xticklabels=xlabels,
            cmap="YlGnBu"
        )
        plt.draw()

    def secplot(self, data, anote=True):
        plt.clf()
        self.plot(data, anote)
        plt.show()


class CM(FigureCanvas):
    # Confusion matrix for classification model
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(data, anote=False):
        plt.rc('font', family='Arial')
        xlabels = data.index.values.tolist()
        ylabels = data.columns.values.tolist()
        sns.heatmap(
            data,
            annot=anote,
            square=False,
            yticklabels=ylabels,
            xticklabels=xlabels,
            cmap="YlGnBu",
        )
        plt.draw()

    def secplot(self, data, anote=True):
        plt.clf()
        self.plot(data, anote=anote)
        plt.show()


class PcCOV(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.21875, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(data, anote=False):
        plt.rc('font', family='Arial')
        xlabels = data.index.values.tolist()
        ylabels = data.columns.values.tolist()
        sns.heatmap(
            data,
            annot=anote,
            yticklabels=ylabels,
            xticklabels=xlabels,
        )
        plt.draw()

    def secplot(self, data, anote):
        plt.clf()
        self.plot(data, anote=anote)
        plt.show()


class TXT(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(5, 4), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(data):
        plt.rc('font', family='Arial')
        x = data['features'].values.tolist()
        y = data['importance'].values.tolist()
        plt.bar(x, height=y, width=0.5, alpha=0.6, color='orangered')
        plt.legend(["Importance"])

    def secplot(self, data):
        plt.clf()
        self.plot(data)
        plt.show()


class TrCom(FigureCanvas):
    # Comparison figure for ML training
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(predictions, actual_values, tr_index):
        min_val_list = min(predictions)
        min_val = min_val_list[0]
        max_val_list = max(actual_values)
        max_val = max_val_list[0]
        plt.rc('font', family='Arial')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.xlim([min_val, max_val])
        plt.ylim([min_val, max_val])
        plt.grid(True)
        plt.scatter(actual_values, predictions, c='tomato')
        plt.legend(["Training Prediction"])
        x = [min_val, max_val]
        plt.plot(x, x, lw=0.8, color='black')
        plt.subplots_adjust(bottom=0.2)
        plt.title("R-square:{r2} MAE:{MAE}".format(r2=tr_index['R-square'], MAE=tr_index['MAE']))

    '''
    def secplot(self, predictions, actual_values):
        plt.clf()
        self.plot(predictions, actual_values)
        plt.show()
    '''


class ElCom(FigureCanvas):
    # Evaluation comparison figure for ML evaluation
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(predictions, actual_values, el_index, error_bar=None):
        min_val = min(predictions)
        max_val = max(actual_values)
        plt.rc('font', family='Arial')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.xlim([min_val, max_val])
        plt.ylim([min_val, max_val])
        plt.grid(True)
        if error_bar is not None:
            plt.errorbar(actual_values, predictions, yerr=error_bar.values.ravel().tolist(),
                         fmt=',C9', elinewidth=0.35, ecolor='deepskyblue')
        plt.scatter(actual_values, predictions, c='deepskyblue')
        plt.legend(["Cross Validation Prediction"])
        x = [min_val, max_val]
        plt.plot(x, x, lw=0.8, color='black')
        plt.subplots_adjust(bottom=0.2)
        plt.title("R-square:{r2}±{r2std} MAE:{MAE}±{MAEstd}".format(r2=el_index['R-square'],
                                                                    r2std=el_index['R-square-std'],
                                                                    MAE=el_index['MAE'],
                                                                    MAEstd=el_index['MAE-std']))
    '''
    def secplot(self, predictions, actual_values, error_bar=None):
        plt.clf()
        self.plot(predictions, actual_values, error_bar)
        plt.show()
    '''


class LC(FigureCanvas):
    #
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(validation_report, ylim=None):
        plt.rc('font', family='Arial')
        # title = r"Learning Curves"
        train_sizes = validation_report['train_size']
        train_scores = validation_report['train_scores']
        test_scores = validation_report['test_scores']
        # plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="deepskyblue"
        )
        plt.plot(
            train_sizes,
            train_scores_mean,
            'o-',
            color="tomato",
            label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            'o-',
            color="deepskyblue",
            label="Cross-validation score"
        )
        plt.legend(loc="best")

    def secplot(self, validation_report, ylim=None):
        plt.clf()
        self.plot(validation_report, ylim=ylim)
        plt.show()


class OPR(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(record_dict):
        plt.rc('font', family='Arial')
        x = record_dict['number of iteration']
        y = record_dict['loss']
        plt.xlabel('Numbers Of Iterations')
        plt.ylabel('Loss')
        plt.xlim([-0.5, len(x)-0.5])
        plt.grid(True)
        plt.bar(x, y, width=1, color='deepskyblue', edgecolor='black', linewidth=0.1)
        plt.subplots_adjust(bottom=0.2)

    def secplot(self,  record_dict):
        plt.clf()
        self.plot(record_dict)
        plt.show()


class TrROC(FigureCanvas):
    # Roc curve for classification tasks
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(record_dict, n_classes, lw=2):
        plt.rc('font', family='Arial')
        fpr = record_dict['fpr']
        tpr = record_dict['tpr']
        roc_auc = record_dict['roc_auc']

        # Plot all ROC curves
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle="--",
            linewidth=2,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )
        plot_roc_dup(lw=lw)

    def secplot(self,  record_dict, n_classes):
        plt.clf()
        self.plot(record_dict, n_classes)
        plt.show()


class VC(FigureCanvas):
    # Validation curve for parameters optimization
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(param_range, record_dict, param_name):
        train_scores = record_dict['train_scores']
        test_scores = record_dict['test_scores']
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.rc('font', family='Arial')
        plt.xlabel(param_name)
        plt.ylabel("Score")
        # plt.ylim(0.0, 1.1)
        lw = 2
        plt.grid(True)
        plt.plot(
            param_range,
            train_scores_mean,
            label="Training score",
            color="deepskyblue",
            lw=lw
        )
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2,
            color="deepskyblue",
            lw=lw
        )
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="tomato", lw=lw)
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.2,
            color="tomato",
            lw=lw
        )
        plt.legend(loc="best")

    def secplot(self,  param_range, record_dict, param_name):
        plt.clf()
        self.plot(param_range, record_dict, param_name)
        plt.show()


class ElROC(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(record_dict, lw=2):
        plt.rc('font', family='Arial')
        macro_fpr = record_dict['macro_fpr']
        macro_tpr = record_dict['macro_tpr']
        macro_roc_auc = record_dict['macro_roc_auc']
        micro_fpr = record_dict['micro_fpr']
        micro_tpr = record_dict['micro_tpr']
        micro_roc_auc = record_dict['micro_roc_auc']
        n_times = record_dict['times']
        # Plot all ROC curves
        colors = cycle(["palevioletred",
                        "deeppink",
                        "salmon",
                        "lightcoral",
                        "tomato",
                        "coral",
                        "firebrick",
                        "orangered",
                        "lightsalmon",
                        "indianred"
                        ])
        for i, color in zip(range(n_times), colors):
            plt.plot(
                micro_fpr[i],
                micro_tpr[i],
                color=color,
                lw=lw,
                label="No {0} Micro ROC curve(area = {1:0.2f})".format(i+1, micro_roc_auc[i]),
            )
        colors = cycle(["deepskyblue",
                        "skyblue",
                        "lightskyblue",
                        "steelblue",
                        "dodgerblue",
                        "blue",
                        "darkslateblue",
                        "cornflowerblue",
                        "mediumslateblue",
                        "royalblue"
                        ])
        for i, color in zip(range(n_times), colors):
            plt.plot(
                macro_fpr[i],
                macro_tpr[i],
                color=color,
                lw=lw,
                label="No {0} Macro ROC curve(area = {1:0.2f})".format(i+1, macro_roc_auc[i]),
            )
        plot_roc_dup(lw=lw)
        # plt.legend(loc="best")

    def secplot(self,  record_dict):
        plt.clf()
        self.plot(record_dict)
        plt.show()


class EpVAR(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        self.ax1 = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)

    def plot(self, validation_report, ylabel='Explained Variance'):
        x_r = validation_report['scores_x']
        exp = validation_report['explained_variance']
        exp_ratio = validation_report['explained_variance_ratio']
        x = []
        for i in range(len(x_r)):
            x.append(str(i+1))
        cumulative = []
        for i in range(len(exp_ratio)):
            temp_cumulative = exp_ratio[i]
            for j in range(i):
                temp_cumulative += exp_ratio[j]
            cumulative.append(temp_cumulative)
        self.ax1.bar(
            x,
            exp,
            label=ylabel,
            alpha=0.7,
            width=0.45,
            color='grey',
        )
        plt.xlabel('Numbers of Principle Components')
        plt.ylabel(ylabel)
        ax2 = self.ax1.twinx()
        ax2.plot(
            x,
            exp_ratio,
            label='{value} Ratio'.format(value=ylabel),
            color='gold',
            lw=2,
            marker='o',
            markersize=5
        )  # 设置线粗细，节点样式
        ax2.plot(
            x,
            cumulative,
            label='Cumulative',
            color='tomato',
            lw=2,
            marker='o',
            markersize=5
        )  # 设置线粗细，节点样式
        plt.yticks()
        # ax1.legend(loc="upper left", shadow=False, scatterpoints=1)
        ax2.legend(loc="best", shadow=False, scatterpoints=1)

    # def secplot(self, validation_report):
        # self.__init__()
        # self.plot(validation_report)
        # plt.show()

    @staticmethod
    def triplot():
        plt.show()


# 带标签的2d对比图
class US2D(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100, tight_layout=True)
        FigureCanvas.__init__(self, fig)

    @staticmethod
    def plot(x_column, y_column, pd_features, pd_class):
        l_column = pd_class.columns.values.tolist()[0]
        x = pd_features[x_column]
        y = pd_features[y_column]
        xy = pd.concat([x, y], axis=1)
        xyl = pd.concat([xy, pd_class], axis=1)
        class_list = xyl[l_column].unique()
        for temp_class_information in class_list:
            temp_xyl = xyl[xyl[l_column].isin([temp_class_information])]
            temp_x = temp_xyl[x_column]
            temp_y = temp_xyl[y_column]
            plt.scatter(x=temp_x, y=temp_y, label=temp_class_information)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend(loc="best", shadow=False, scatterpoints=1)

    def secplot(self, x_column, y_column, pd_features, pd_class):
        plt.clf()
        self.plot(x_column, y_column, pd_features, pd_class)
        plt.show()


# 带标签的3d对比图
class US3D(FigureCanvas):
    def __init__(self):
        fig = plt.figure(figsize=(4.5, 3.375), dpi=100)
        self.ax1 = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(self.ax1)
        FigureCanvas.__init__(self, fig)

    def plot(self, x_column, y_column, z_column, pd_features, pd_class):
        l_column = pd_class.columns.values.tolist()[0]
        x = pd_features[x_column]
        y = pd_features[y_column]
        z = pd_features[z_column]
        xy = pd.concat([x, y], axis=1)
        xyz = pd.concat([xy, z], axis=1)
        xyzl = pd.concat([xyz, pd_class], axis=1)
        class_list = xyzl[l_column].unique()
        for temp_class_information in class_list:
            temp_xyzl = xyzl[xyzl[l_column].isin([temp_class_information])]
            temp_x = temp_xyzl[x_column].values  # .T.drop_duplicates().T.values
            temp_y = temp_xyzl[y_column].values  # .T.drop_duplicates().T.values
            temp_z = temp_xyzl[z_column].values  # .T.drop_duplicates().T.values
            self.ax1.scatter(
                temp_x,
                temp_y,
                temp_z,
                label=temp_class_information
            )
        self.ax1.set_xlabel(x_column)
        self.ax1.set_ylabel(y_column)
        self.ax1.set_zlabel(z_column)
        plt.legend(loc="best", shadow=False, scatterpoints=1)

    # def secplot(self, x_column, y_column, z_column, pd_features, pd_class):
    #    plt.clf()
    #    self.plot(x_column, y_column, z_column, pd_features, pd_class)
    #    plt.show()
