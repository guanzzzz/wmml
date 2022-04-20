from subwin.wmmlsubeng import *
from subwin.wmmlprediction import Ui_WMML_prediction
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from MLparameters import *
import sys


class WMML_pred(QWidget, Ui_WMML_prediction):
    """
    WMML train and score widget
    available Input signal:
    data: features, targets and relative metas, type: array-like object
    model: Model object
    """

    def __init__(self, md_file=None):
        super(WMML_pred, self).__init__()
        self.setupUi(self)
        self.pd_model = Model()
        self.jzdst.clicked.connect(self.reload_data)
        self.jzrld_btn.clicked.connect(self.reload_model)
        self.pd_model = self.pd_features = self.pd_ft_md = self.pd_pd_md = self.pd_original_dataset = None
        self.pd_ds_md = self.tr_targets = self.tr_metas = None
        if md_file is not None:
            self.pd_model = load_model(md_file)
            self.pd_param_tab1.clear()
            dict_to_browse_on_tab1 = {'Model Name': self.pd_model.model_names}
            for key, value in dict_to_browse_on_tab1.items():
                self.pd_param_tab1.append('{key}:{value}'.format(key=key, value=value))
            self.pd_param_tab2.clear()
            k = 1
            for i in self.pd_model.tr_header:
                self.pd_param_tab2.append('No{key} Features: {value}'.format(key=k, value=i))
                k += 1
            self.infobox.setText("Model Loaded Complete")

    @pyqtSlot()
    def on_jzatm_btn_clicked(self):
        if self.pd_model is None:
            pass
        else:
            try:
                temp_df = pd.DataFrame()
                for i in self.pd_model.tr_header:
                    temp_df_2 = self.pd_original_dataset.loc[
                                :, self.pd_original_dataset.columns.intersection([i])]
                    temp_df = pd.concat([temp_df, temp_df_2], axis=1)
                self.pd_features = temp_df
                self.pd_ft_md = table_view(self.pd_fd_tab, self.pd_features)
                self.infobox.setText("Auto Match Complete")
            except Exception as e:
                self.infobox.setText("Feature Mismatch")
                print(e)

    @pyqtSlot()
    def on_pd_pred_btn_clicked(self):
        try:
            self.pd_model.pd_pd = pd.DataFrame(data=self.pd_model.model.predict(self.pd_features.values),
                                               columns=['prediction'])
            self.pd_pd_md = table_view(self.pd_pd_tab, self.pd_model.pd_pd)
            self.infobox.setText("Prediction Complete!")
        except Exception as e:
            self.infobox.setText("Prediction Failed!")
            print(e)

    @pyqtSlot()
    def on_jzsdd_clicked(self):
        if self.pd_model.pd_pd is None:
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
                    self.pd_model.pd_pd.to_csv(path_save_position + '.csv')
            else:
                pass

    def reload_data(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'csv files( *.csv)',
                                                        options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                if path_data_position == '':
                    pass
                else:
                    self.pd_original_dataset = pd.read_csv(path_data_position)
                    self.pd_ds_md = table_view(self.jz_od_tab, self.pd_original_dataset)
                    self.infobox.setText(path_data_position)
        except Exception as e:
            print(e)

    def reload_model(self):
        try:
            file_diag = QFileDialog()
            data_position = QFileDialog.getOpenFileName(file_diag, 'select file', '', 'pkcls files(*.pkcls)',
                                                        options=options)
            if isinstance(data_position, tuple):
                path_data_position = data_position[0]
                if path_data_position == '':
                    pass
                else:
                    self.pd_model = load_model(path_data_position)
                    self.pd_param_tab1.clear()
                    dict_to_browse_on_tab1 = {'Model Name': self.pd_model.model_names}
                    for key, value in dict_to_browse_on_tab1.items():
                        self.pd_param_tab1.append('{key}:{value}'.format(key=key, value=value))
                    self.pd_param_tab2.clear()
                    k = 1
                    for i in self.pd_model.tr_header:
                        self.pd_param_tab2.append('No{key} Features: {value}'.format(key=k, value=i))
                        k += 1
                    self.infobox.setText("Model Loaded Complete")
        except Exception as e:
            print(e)

    def get_data_from_sender(self, features, metas=None, targets=None):
        if metas is not None:
            self.tr_targets = targets
            self.tr_metas = metas
        self.pd_original_dataset = features
        self.pd_ds_md = table_view(self.jz_od_tab, self.pd_original_dataset)

    def get_model_from_sender(self, model):
        self.pd_model = model
        self.pd_param_tab1.clear()
        dict_to_browse_on_tab1 = {'Model Name': self.pd_model.model_names}
        for key, value in dict_to_browse_on_tab1.items():
            self.pd_param_tab1.append('{key}:{value}'.format(key=key, value=value))
        self.pd_param_tab2.clear()
        k = 1
        for i in self.pd_model.tr_header:
            self.pd_param_tab2.append('No {key} Features:{value}'.format(key=k, value=i))
            k += 1
        self.infobox.setText("Model Loaded Complete")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WMML_pred()
    win.show()
    sys.exit(app.exec_())
