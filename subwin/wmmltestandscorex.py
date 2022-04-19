from subwin.wmmltestandscore import Ui_WMML_train
from subwin.wmmlsubeng import *
from subwin.wmmlknnx import WMML_knn
from subwin.wmmlgbdtx import WMML_gbdt
from subwin.wmmlrfx import WMML_rf
from subwin.wmmlnetx import WMML_nn
from subwin.wmmlstkx import WMML_stk
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.utils import shuffle
from sklearn import metrics, model_selection, neighbors, ensemble, neural_network
from hyperopt import STATUS_OK, Trials, fmin, tpe, rand, anneal
from MLparameters import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class TrainThread(QThread):
    """
    Model training thread
    Input:
    features: training features, type: array_like object
    targets: training targets, type: array_like object
    model: Model object
    Output signal:
    model: Model object with trained ML model and relative information
    str: string to be viewed in infobox
    """
    Output_signal = pyqtSignal(object, str)

    def __init__(self, features, targets, model):
        super().__init__()
        self.tr_features = features
        self.tr_targets = targets
        self.ml_model = model
        self.e = None

    def run(self):
        try:
            self.ml_model.model.fit(self.tr_features.values, self.tr_targets.values.ravel())
            temp_array = self.ml_model.model.predict(self.tr_features)
            self.ml_model.tr_pd = pd.DataFrame(data=temp_array, columns=['predicton']).round(decimals=6)
            self.ml_model.tr_header = self.tr_features.columns.values.tolist()
            r2 = metrics.r2_score(self.tr_targets, self.ml_model.tr_pd)
            self.ml_model.tr_index['R-square'] = round(r2, 4)
            mae = metrics.mean_absolute_error(self.tr_targets, self.ml_model.tr_pd)
            self.ml_model.tr_index['MAE'] = round(mae, 4)
            mse = metrics.mean_squared_error(self.tr_targets, self.ml_model.tr_pd)
            self.ml_model.tr_index['MSE'] = round(mse, 4)
            self.Output_signal.emit(self.ml_model, 'Finished Training')
        except Exception as e:
            self.e = e
            self.Output_signal.emit(None, str(e))


class OptThread(QThread):
    """
    Model Optimization thread
    Input:
    features: training features, type: array_like object
    targets: training targets, type: array_like object
    model: Model object with defined optimization parameters
    Output signal:
    model: Model object with redefined ML model with optimized parameters and relative information
    str: string to be viewed in infobox
    """
    Output_signal = pyqtSignal(object, str)

    def __init__(self, features, targets, model):
        super().__init__()
        self.tr_features = features
        self.tr_targets = targets
        self.ml_model = model
        self.e = None

    def run(self):
        best = {}
        trials = Trials()
        if self.ml_model.model_names == "Random Forest":
            def define_opt_model(params):
                clf = ensemble.RandomForestRegressor(**params)
                scores = model_selection.cross_val_score(
                    clf,
                    self.tr_features.values,
                    self.tr_targets.values.ravel(),
                    cv=self.ml_model.opt_folds
                )
                self.Output_signal.emit(None, 'Current Validation Score:{}'.format(np.mean(scores)))
                return {'loss': 1 - np.mean(scores),
                        'status': STATUS_OK}

            if self.ml_model.opt_methods == "Random":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=rand.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            elif self.ml_model.opt_methods == "Adaptive":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=anneal.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            else:
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=tpe.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            if 'criterion' in self.ml_model.opt_compare_space:
                best['criterion'] = self.ml_model.opt_compare_space['criterion'][best['criterion']]
            if 'bootstrap' in self.ml_model.opt_compare_space:
                best['bootstrap'] = self.ml_model.opt_compare_space['bootstrap'][best['bootstrap']]
            if 'max_features' in self.ml_model.opt_compare_space:
                best['max_features'] = self.ml_model.opt_compare_space['max_features'][best['max_features']]
            self.ml_model = WMML_rf.redefine_model(self.ml_model)
        elif self.ml_model.model_names == "K-Neighbors":
            def define_opt_model(params):
                clf = neighbors.KNeighborsRegressor(**params)
                scores = model_selection.cross_val_score(
                    clf,
                    self.tr_features.values,
                    self.tr_targets.values.ravel(),
                    cv=self.ml_model.opt_folds
                )
                self.Output_signal.emit(None, 'Current Validation Score:{}'.format(np.mean(scores)))
                return {'loss': 1 - np.mean(scores),
                        'status': STATUS_OK}

            if self.ml_model.opt_methods == "Random":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=rand.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            elif self.ml_model.opt_methods == "Adaptive":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=anneal.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            else:
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=tpe.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            if 'algorithm' in self.ml_model.opt_compare_space:
                best['algorithm'] = self.ml_model.opt_compare_space['algorithm'][best['algorithm']]
            if 'metric' in self.ml_model.opt_compare_space:
                best['metric'] = self.ml_model.opt_compare_space['metric'][best['metric']]
            if 'weights' in self.ml_model.opt_compare_space:
                best['weights'] = self.ml_model.opt_compare_space['weights'][best['weights']]
            self.ml_model = WMML_knn.redefine_model(self.ml_model)
        elif self.ml_model.model_names == "Neural Network":
            def define_opt_model(params):
                clf = neural_network.MLPRegressor(
                    **params,
                    alpha=self.ml_model.parameters['alpha'],
                    early_stopping=self.ml_model.parameters['early_stopping'],
                    validation_fraction=self.ml_model.parameters['validation_fraction']
                )
                scores = model_selection.cross_val_score(
                    clf,
                    self.tr_features.values,
                    self.tr_targets.values.ravel(),
                    cv=self.ml_model.opt_folds
                )
                self.Output_signal.emit(None, 'Current Validation Score:{}'.format(np.mean(scores)))
                return {'loss': 1 - np.mean(scores),
                        'status': STATUS_OK}

            if self.ml_model.opt_methods == "Random":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=rand.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            elif self.ml_model.opt_methods == "Adaptive":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=anneal.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            else:
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=tpe.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            if 'hidden_layer_sizes' in self.ml_model.opt_compare_space:
                best['hidden_layer_sizes'] = self.ml_model.opt_compare_space[
                    'hidden_layer_sizes'][best['hidden_layer_sizes']]
            if 'activation' in self.ml_model.opt_compare_space:
                best['activation'] = self.ml_model.opt_compare_space['activation'][best['activation']]
            if 'solver' in self.ml_model.opt_compare_space:
                best['solver'] = self.ml_model.opt_compare_space['solver'][best['solver']]
            self.ml_model = WMML_nn.redefine_model(self.ml_model)
        elif self.ml_model.model_names == "Gradient Boosting":
            def define_opt_model(params):
                clf = ensemble.GradientBoostingRegressor(random_state=self.ml_model.parameters['random_state'],
                                                         **params)
                scores = model_selection.cross_val_score(
                    clf,
                    self.tr_features.values,
                    self.tr_targets.values.ravel(),
                    cv=self.ml_model.opt_folds
                )
                self.Output_signal.emit(None, 'Current Validation Score:{}'.format(np.mean(scores)))
                return {'loss': 1 - np.mean(scores),
                        'status': STATUS_OK}

            if self.ml_model.opt_methods == "Random":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=rand.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            elif self.ml_model.opt_methods == "Adaptive":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=anneal.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            else:
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=tpe.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            if 'init' in self.ml_model.opt_compare_space:
                best['init'] = self.ml_model.opt_compare_space['init'][best['init']]
            if 'loss' in self.ml_model.opt_compare_space:
                best['loss'] = self.ml_model.opt_compare_space['loss'][best['loss']]
            if 'criterion' in self.ml_model.opt_compare_space:
                best['criterion'] = self.ml_model.opt_compare_space['criterion'][best['criterion']]
            if 'max_features' in self.ml_model.opt_compare_space:
                best['max_features'] = self.ml_model.opt_compare_space['max_features'][best['max_features']]
            self.ml_model = WMML_gbdt.redefine_model(self.ml_model)
        elif self.ml_model.model_names == "Stacking Model":
            def define_opt_model(params):
                clf = ensemble.StackingRegressor(
                    estimators=self.ml_model.parameters['estimators'],
                    **params
                )
                scores = model_selection.cross_val_score(
                    clf,
                    self.tr_features.values,
                    self.tr_targets.values.ravel(),
                    cv=self.ml_model.opt_folds
                )
                self.Output_signal.emit(None, 'Current Validation Score:{}'.format(np.mean(scores)))
                return {'loss': 1 - np.mean(scores),
                        'status': STATUS_OK}

            if self.ml_model.opt_methods == "Random":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=rand.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            elif self.ml_model.opt_methods == "Adaptive":
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=anneal.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            else:
                best = fmin(
                    define_opt_model,
                    space=self.ml_model.opt_space,
                    algo=tpe.suggest,
                    max_evals=self.ml_model.opt_times,
                    trials=trials
                )
            if 'passthrough' in self.ml_model.opt_compare_space:
                best['passthrough'] = self.ml_model.opt_compare_space['passthrough'][best['passthrough']]
            self.ml_model = WMML_stk.redefine_model(self.ml_model)
        self.ml_model.parameters.update(best)
        self.ml_model.opt_record = {
            'number of iteration': [x['tid'] for x in trials.trials],
            'loss': [y['result']['loss'] for y in trials.trials]
        }
        self.Output_signal.emit(self.ml_model, 'Optimization Complete')


class ValidThread(QThread):
    """
    Model Optimization thread
    Input:
    features: training features, type: array_like object
    targets: training targets, type: array_like object
    model: Model object with defined optimization space
    score_indices: evaluation methods, only support R-square for this limited version
    Output signal:
    model: Model object with validation curve information
    data: the parameters with r-square
    str: string to be viewed in infobox
    """
    Output_signal = pyqtSignal(object, object, str)

    def __init__(self, features, targets, model, score_indices):
        super().__init__()
        self.tr_features = features
        self.tr_targets = targets
        self.ml_model = model
        self.score_indices = score_indices
        self.e = None

    def run(self):
        train_scores, test_scores = model_selection.validation_curve(
            self.ml_model.model,
            self.tr_features.values,
            self.tr_targets,
            param_name=self.ml_model.temp_opt_param_name,
            param_range=self.ml_model.opt_valid_space[self.ml_model.temp_opt_param_name],
            scoring=self.score_indices,
            cv=self.ml_model.opt_valid_folds
        )
        self.ml_model.opt_valid_data["train_scores"] = train_scores
        self.ml_model.opt_valid_data["test_scores"] = test_scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_mean = np.round(train_scores_mean, 4)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_mean = np.round(test_scores_mean, 4)
        temp_sct_df = pd.DataFrame(data=train_scores_mean, columns=["train_scores"])
        temp_scv_df = pd.DataFrame(data=test_scores_mean, columns=["test_scores"])
        temp_param_df = pd.DataFrame(
            data=self.ml_model.opt_valid_space[self.ml_model.temp_opt_param_name],
            columns=[str(self.ml_model.temp_opt_param_name)]
        )
        temp_df1 = pd.concat([temp_sct_df, temp_scv_df], axis=1)
        temp_df2 = pd.concat([temp_df1, temp_param_df], axis=1)
        self.Output_signal.emit(self.ml_model, temp_df2, 'Calculation Complete')


class EvlThread(QThread):
    """
    Model Optimization thread
    Input:
    features: training features, type: array_like object
    targets: training targets, type: array_like object
    model: Model object with defined model parameters
    parameters: evaluation parameters
    Output signal:
    model: Model object with validation curve information
    data: evaluation parameters
    str: string to be viewed in infobox
    """
    Output_signal = pyqtSignal(object, object, str)
    Meta_signal = pyqtSignal(str)

    def __init__(self, features, targets, model, parameters):
        super().__init__()
        self.tr_features = features
        self.tr_targets = targets
        self.ml_model = model
        self.parameters = parameters
        self.e = None

    def run(self):
        self.ml_model.el_pd = pd.DataFrame()
        self.ml_model.el_pd_all = pd.DataFrame()
        self.ml_model.el_pd_error = None
        # Cross Validation prediction
        r2_list = []
        mae_list = []
        mse_list = []
        y_true_columns_list = []
        y_pred_columns_list = []
        temp_df = pd.DataFrame()
        # Set repeat times
        for i in range(self.parameters['n_repeats']):
            temp_tr_features_with_idx = pd.concat([self.tr_features,
                                                   pd.DataFrame(data=self.tr_features.index.values, columns=["idx"])],
                                                  axis=1)
            if self.parameters['shuffle']:
                x_test_with_idx, y_test = shuffle(
                    temp_tr_features_with_idx,
                    self.tr_targets.values.ravel(),
                    random_state=self.parameters['random_state']
                )
            else:
                x_test_with_idx, y_test = temp_tr_features_with_idx, self.tr_targets.values.ravel()
            temp_header = self.tr_features.columns.values.tolist()
            temp_header.append("idx")
            temp_x_test_with_idx = pd.DataFrame(data=x_test_with_idx, columns=temp_header)
            temp_index_header = temp_header[-1]
            idx_df = temp_x_test_with_idx["idx"]
            x_test = temp_x_test_with_idx.drop(temp_index_header, axis=1, inplace=False)
            cv_pred = model_selection.cross_val_predict(self.ml_model.model, x_test, y_test,
                                                        cv=self.parameters['n_splits'])
            y_true_columns_list.append("No{} Actual Value".format(i + 1))
            y_pred_columns_list.append("No{} Predicted Value".format(i + 1))
            r2s = metrics.r2_score(y_test, cv_pred)
            r2_list.append(r2s)
            mae_list.append(metrics.mean_absolute_error(y_test, cv_pred))
            mse_list.append(metrics.mean_squared_error(y_test, cv_pred))
            temp_df4 = pd.concat(
                [pd.DataFrame(data=y_test, columns=[y_true_columns_list[i]]),
                 pd.DataFrame(data=cv_pred, columns=[y_pred_columns_list[i]]),
                 idx_df], axis=1
            )
            temp_df5 = temp_df4.sort_values(by="idx")
            temp_df6 = temp_df5.drop("idx", axis=1, inplace=False)
            self.ml_model.el_pd_all = pd.concat([self.ml_model.el_pd_all, temp_df6], axis=1)
            temp_df = pd.concat([temp_df, temp_df6], axis=1)
            self.Meta_signal.emit("No{pp}-{nnnm}Fold R-square:{r2s2}".format(pp=i, nnnm=self.parameters['n_splits'],
                                                                             r2s2=r2s))
        # Calculated the mean and error
        cv_pred_list = []
        cv_pred_std_list = []
        for indexs in temp_df.index:
            temp_list = temp_df.loc[indexs].values[:]
            cv_pred_list.append(np.mean(temp_list))
            cv_pred_std_list.append(np.std(temp_list))
        self.ml_model.el_pd = pd.concat([pd.DataFrame(data=self.tr_targets.values, columns=['Actual Value']),
                                         pd.DataFrame(data=cv_pred_list, columns=['Predicted Value'])], axis=1)
        self.ml_model.el_pd_all_err = pd.DataFrame(cv_pred_std_list, columns=['Std'])
        # Calculate the evaluation indices
        r2_array = np.array(r2_list)
        mae_array = np.array(mae_list)
        mse_array = np.array(mse_list)
        r2_mean = np.mean(r2_array)
        self.ml_model.eval_index["R-square"] = round(r2_mean, 4)
        r2_std = np.std(r2_array, ddof=1)
        self.ml_model.eval_index["R-square-std"] = round(r2_std, 4)
        mae_mean = np.mean(mae_array)
        self.ml_model.eval_index["MAE"] = round(mae_mean, 4)
        mae_std = np.std(mae_array, ddof=1)
        self.ml_model.eval_index["MAE-std"] = round(mae_std, 4)
        mse_mean = np.mean(mse_array)
        self.ml_model.eval_index["MSE"] = round(mse_mean, 4)
        mse_std = np.std(mse_array, ddof=1)
        self.ml_model.eval_index["MSE-std"] = round(mse_std, 4)
        self.Output_signal.emit(self.ml_model, self.parameters, 'Evaluation Complete')


class WMML_tas(QWidget, Ui_WMML_train):
    """
    WMML train and score widget
    available Input signal:
    data: features, targets and relative metas, type: array-like object
    model: Model object
    available Output signal:
    model: Model object
    """
    model_signal = pyqtSignal(object)

    def __init__(self):
        super(WMML_tas, self).__init__()
        self.setupUi(self)
        self.ml_model = Model()
        self.tr_features = self.tr_targets = self.tr_metas = pd.DataFrame()
        self.opt_bgtsk = self.vld_bgtsk = self.tr_bgtsk = self.el_bgtsk = None
        self.cv_rand.setValidator(QIntValidator())
        self.cv_rand.setText('None')
        self.mp_md = self.fd_model = self.td_model = self.tr_pd_md = None
        self.ml_tr_fig = self.ml_tr_com_fig = self.opt_pm_md = self.opt_rc_md = None
        self.ml_opt_fig = self.ml_opt_rec_fig = self.ml_opt_fig_2 = self.ml_vc_fig = None
        self.ml_el_fig = self.ml_el_com_fig = self.ml_el_fig_tbr = self.el_pd_md = None

    def ml_info_browse(self, model):
        table_to_browse_on_tab1 = pd.DataFrame([model.parameters])
        self.mp_md = table_view(self.ml_param_tab, table_to_browse_on_tab1)
        self.ml_param_tab2.clear()
        dict_to_browse_on_tab2 = {}
        if model.opt_selection:
            dict_to_browse_on_tab2['Optimization'] = "Enabled"
            dict_to_browse_on_tab2['Algorithm'] = model.opt_methods
            dict_to_browse_on_tab2['Repeat Times'] = model.opt_times
        else:
            dict_to_browse_on_tab2['Optimization'] = "Disabled"
        for key, value in dict_to_browse_on_tab2.items():
            self.ml_param_tab2.append('{key}:{value}'.format(key=key, value=value))
        dict_to_browse_on_tab3 = {'Model Name': model.model_names}
        self.ml_param_tab3.clear()
        for key, value in dict_to_browse_on_tab3.items():
            self.ml_param_tab3.append('{key}:{value}'.format(key=key, value=value))

    def get_data_from_sender(self, features, metas=None, targets=None):
        if metas is not None:
            self.tr_targets = targets
            self.tr_metas = metas
        self.tr_features = features
        self.fd_model = table_view(self.jz_fd_tab, self.tr_features)
        self.td_model = table_view(self.jz_td_tab, self.tr_targets)

    def get_model_from_sender(self, model):
        if model is not None:
            self.ml_model = model
            self.ml_info_browse(self.ml_model)
        self.hss_csc.clear()
        self.hss_csc.addItem("")
        if self.ml_model.opt_selection:
            for name in self.ml_model.opt_valid_space.keys():
                self.hss_csc.addItem(name)
        else:
            pass

    @pyqtSlot()
    def on_ml_tr_bg_btn_clicked(self):
        if self.ml_model.model is None:
            self.infobox.setText("No Active Model Detected")
        elif self.tr_features.empty:
            self.infobox.setText("No Active Features Detected")
        elif self.tr_targets.empty:
            self.infobox.setText("No Active Targets Detected")
        else:
            try:
                if self.tr_bgtsk is None:
                    self.tr_bgtsk = TrainThread(self.tr_features, self.tr_targets, self.ml_model)
                    self.tr_bgtsk.Output_signal.connect(self.train_complete)
                    self.tr_bgtsk.start()
                    self.infobox.setText("Begin Training....")
                elif self.tr_bgtsk.isRunning():
                    self.infobox.setText("Last Training Task Is Still Running")
                else:
                    self.tr_bgtsk = TrainThread(self.tr_features, self.tr_targets, self.ml_model)
                    self.tr_bgtsk.Output_signal.connect(self.train_complete)
                    self.tr_bgtsk.start()
                    self.infobox.setText("Begin Training....")
            except Exception as e:
                self.infobox.setText(str(e))

    def train_complete(self, model, strv):
        if model is not None:
            self.ml_model = model
            self.tr_pd_md = table_view(self.tr_pd_tab, self.ml_model.tr_pd)
            self.ml_tr_fig = QGridLayout()
            self.ml_tr_fig.setObjectName("ml_tr_fig")
            self.gridLayout_9.addLayout(self.ml_tr_fig, 0, 0, 1, 1)
            self.ml_tr_com_fig = TrCom()
            self.ml_tr_com_fig.plot(self.ml_model.tr_pd.values.tolist(), self.tr_targets.values.tolist(),
                                    self.ml_model.tr_index)
            self.ml_tr_fig.addWidget(self.ml_tr_com_fig)
            self.ml_tr_fig.addWidget(NavigationToolbar(self.ml_tr_com_fig, self))
            self.ml_param_tab4.clear()
            self.ml_param_tab4.append('Features Used')
            k = 1
            for i in self.ml_model.tr_header:
                self.ml_param_tab4.append('No {key} Feature:{value}'.format(key=k, value=i))
                k += 1
            self.infobox.setText(strv)
        else:
            self.infobox.setText(strv)

    @pyqtSlot()
    def on_ml_op_bg_btn_clicked(self):
        if self.ml_model.model is None:
            self.infobox.setText("No Active Model Detected")
        elif self.tr_features.empty:
            self.infobox.setText("No Active Features Detected")
        elif self.tr_targets.empty:
            self.infobox.setText("No Active Targets Detected")
        elif self.ml_model.opt_selection is False:
            self.infobox.setText("Optimization Selection Is Not Enabled")
        else:
            try:
                if self.opt_bgtsk is None:
                    self.opt_bgtsk = OptThread(self.tr_features, self.tr_targets, self.ml_model)
                    self.opt_bgtsk.Output_signal.connect(self.opt_complete)
                    self.opt_bgtsk.start()
                    self.infobox.setText("Begin Optimization....")
                elif self.opt_bgtsk.isRunning():
                    self.infobox.setText("Last Optimization Task Is Still Running")
                else:
                    self.opt_bgtsk = OptThread(self.tr_features, self.tr_targets, self.ml_model)
                    self.opt_bgtsk.Output_signal.connect(self.opt_complete)
                    self.opt_bgtsk.start()
                    self.infobox.setText("Begin Optimization....")
            except Exception as e:
                self.infobox.setText(str(e))

    def opt_complete(self, model, strv):
        if model is None:
            self.infobox.setText(strv)
        else:
            self.ml_model = model
            self.ml_info_browse(self.ml_model)
            self.opt_pm_md = table_view(self.ml_param_tab_2, pd.DataFrame([self.ml_model.parameters]))
            self.ml_opt_tab.clear()
            self.ml_opt_tab.append(
                "{} Folds Minimum Loss:{:.4f}".format(
                    self.ml_model.opt_folds,
                    min(self.ml_model.opt_record['loss'])
                )
            )
            self.opt_rc_md = table_view(self.tr_el_tab_2, pd.DataFrame.from_dict(self.ml_model.opt_record))
            self.ml_opt_fig = QGridLayout()
            self.ml_opt_fig.setObjectName("ml_opt_fig")
            self.gridLayout_12.addLayout(self.ml_opt_fig, 0, 0, 1, 1)
            self.ml_opt_rec_fig = OPR()
            self.ml_opt_rec_fig.plot(self.ml_model.opt_record)
            self.ml_opt_fig.addWidget(self.ml_opt_rec_fig)
            # self.ml_opt_fig.addWidget(NavigationToolbar(self.ml_opt_rec_fig, self))
            self.infobox.setText(strv)

    @pyqtSlot()
    def on_hss_compute_clicked(self):
        if self.ml_model.model is None:
            self.infobox.setText("No Active Model Detected")
        elif self.tr_features.empty:
            self.infobox.setText("No Active Features Detected")
        elif self.tr_targets.empty:
            self.infobox.setText("No Active Targets Detected")
        elif self.ml_model.opt_selection is False:
            self.infobox.setText("Optimization Selection Is Not Enabled")
        else:
            self.ml_model.opt_valid_folds = eval(self.hss_fd.currentText())
            self.ml_model.temp_opt_param_name = self.hss_csc.currentText()
            if self.ml_model.model_classes == 'c':
                targets = self.tr_classes.values.ravel()
                score_indices = "accuracy"
            else:
                targets = self.tr_targets.values.ravel()
                score_indices = "r2"

            if self.vld_bgtsk is None:
                self.vld_bgtsk = ValidThread(self.tr_features, targets, self.ml_model, score_indices)
                self.vld_bgtsk.Output_signal.connect(self.vld_complete)
                self.vld_bgtsk.start()
                self.infobox.setText("Begin Calculation....")
            elif self.vld_bgtsk.isRunning():
                self.infobox.setText("Last Calculation Task Is Still Running")
            else:
                self.vld_bgtsk = ValidThread(self.tr_features, targets, self.ml_model, score_indices)
                self.vld_bgtsk.Output_signal.connect(self.vld_complete)
                self.vld_bgtsk.start()
                self.infobox.setText("Begin Calculation....")

    def vld_complete(self, model, df, strv):
        self.ml_model = model
        self.opt_rc_md = table_view(self.tr_el_tab_2, df)
        self.infobox.setText(strv)
        self.ml_opt_fig_2 = QGridLayout()
        self.ml_opt_fig_2.setObjectName("ml_opt_fig")
        self.gridLayout_4.addLayout(self.ml_opt_fig_2, 0, 0, 1, 1)
        self.ml_vc_fig = VC()
        self.ml_vc_fig.plot(
            self.ml_model.opt_valid_space[self.ml_model.temp_opt_param_name],
            self.ml_model.opt_valid_data,
            self.ml_model.temp_opt_param_name
        )
        self.ml_opt_fig_2.addWidget(self.ml_vc_fig)

    @pyqtSlot()
    def on_ml_el_btn_clicked(self):
        if self.ml_model.model is None:
            self.infobox.setText("No Active Model Detected")
        elif self.tr_features.empty:
            self.infobox.setText("No Active Features Detected")
        elif self.tr_targets.empty:
            self.infobox.setText("No Active Targets Detected")
        else:
            parameters = {}
            # selections = "cross_validation"
            if self.cv_sf.isChecked():
                parameters['shuffle'] = True
            else:
                parameters['shuffle'] = False
            try:
                random_state = self.cv_rand.text()
                parameters['random_state'] = eval(random_state)
            except Exception as e:
                parameters['random_state'] = None
                print(str(e))
            fold_cmbox_text = self.cv_f.currentText()
            parameters['n_splits'] = eval(fold_cmbox_text)
            times_cmbox_text = self.vd_tms.currentText()
            parameters['n_repeats'] = eval(times_cmbox_text)

            if self.el_bgtsk is None:
                self.el_bgtsk = EvlThread(self.tr_features, self.tr_targets, self.ml_model, parameters)
                self.el_bgtsk.Meta_signal.connect(self.evl_metas_real_time)
                self.el_bgtsk.Output_signal.connect(self.evl_complete)
                self.el_bgtsk.start()
                self.infobox.setText("Begin Evaluation....")
            elif self.el_bgtsk.isRunning():
                self.infobox.setText("Last Evaluation Task Is Still Running")
            else:
                self.el_bgtsk = EvlThread(self.tr_features, self.tr_targets, self.ml_model, parameters)
                self.el_bgtsk.Meta_signal.connect(self.evl_metas_real_time)
                self.el_bgtsk.Output_signal.connect(self.evl_complete)
                self.el_bgtsk.start()
                self.infobox.setText("Begin Evaluation....")

    def evl_metas_real_time(self, metas):
        self.infobox.setText(metas)

    def evl_complete(self, model, parameters, strv):
        self.ml_model = model
        self.infobox.setText(strv)
        self.ml_el_tab.clear()
        self.ml_el_tab.append('{} Times {} Folds Results:'.format(parameters['n_repeats'], parameters['n_splits']))
        for key, value in self.ml_model.eval_index.items():
            self.ml_el_tab.append('{key}:{value}'.format(key=key, value=value))
        try:
            self.ml_el_fig.removeWidget(self.ml_el_fig_tbr)
        except Exception as e:
            print(e)
        try:
            self.ml_el_fig.removeWidget(self.ml_el_com_fig)
        except Exception as e:
            print(e)
        self.ml_el_fig = QGridLayout()
        self.ml_el_fig.setObjectName("ml_el_fig")
        self.gridLayout_2.addLayout(self.ml_el_fig, 0, 0, 1, 1)
        self.ml_el_com_fig = ElCom()
        self.ml_el_com_fig.plot(
            self.ml_model.el_pd['Predicted Value'].values.tolist(),
            self.ml_model.el_pd['Actual Value'].values.tolist(),
            self.ml_model.eval_index,
            self.ml_model.el_pd_all_err
        )
        self.ml_el_fig.addWidget(self.ml_el_com_fig)
        self.ml_el_fig_tbr = NavigationToolbar(self.ml_el_com_fig, self)
        self.ml_el_fig.addWidget(self.ml_el_fig_tbr)
        self.el_pd_md = table_view(self.tr_el_tab, self.ml_model.el_pd_all)

    @pyqtSlot()
    def on_ml_sve_btn_clicked(self):
        if self.el_pd_md is None:
            self.infobox.setText("No available data detected")
        else:
            file_diag = QFileDialog()
            save_position = QFileDialog.getSaveFileName(file_diag, 'save file', './', 'csv files(*.csv)',
                                                        options=options)
            if isinstance(save_position, tuple):
                path_save_position = save_position[0]
                if path_save_position == '':
                    pass
                else:
                    self.ml_model.el_pd.to_csv(path_save_position+'.csv')
            else:
                pass

    @pyqtSlot()
    def on_ml_tr_sv_btn_clicked(self):
        if self.ml_model is not None:
            try:
                file_diag = QFileDialog()
                data_position = QFileDialog.getSaveFileName(file_diag, 'select file', '',
                                                            'pkl files(*.pkcls)', options=options)
                if isinstance(data_position, tuple):
                    path_save_position = data_position[0]
                    if path_save_position == '':
                        pass
                    else:
                        save_model(self.ml_model, path_save_position)
                        self.infobox.setText("Model Save Complete！")
            except Exception as e:
                self.infobox.setText(str(e))
        else:
            self.infobox.setText("No Active Model Detected")

    @pyqtSlot()
    def on_ml_tr_sd_btn_clicked(self):
        if self.ml_model is not None:
            self.model_signal.emit(self.ml_model)
            self.infobox.setText("Model Transferred For Prediction")
        else:
            self.infobox.setText("No Active Model Detected")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_tas()
    win.show()
    sys.exit(app.exec_())
