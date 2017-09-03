import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from MachineLearningUtils.ModelsUtils import ModelUtils


# from sklearn.model_selection import split_data_to_train_test as split


class LinearModelUtils(ModelUtils):
    """
    this class gets data frame and  linear model and returns dictionary:
     model_info={
            linear_model_name   : "the given linear model name or it type + columns',
            col_list            : '&' seperated values of the columns list,
            actual_lbl          : the name of the lable we wants to predict,
            formula             : the formula from the linear model (if exists)
            rmse                : '',
            rmsle               : '',
            mea                 : '',
            plot_predicted_vs_actual : ax of plot_predicted_vs_actual


     }
    """

    def __init__(self,
                 df,
                 lm=None,
                 lm_name="",
                 predicted_lbl=None,
                 actual_lbl=None,
                 columns_lst=[],
                 test_size=0.3,
                 random_state=123456):
        ModelUtils.__init__(
            self,
            df=df,
            model=lm,
            model_name=lm_name,
            predicted_lbl=predicted_lbl,
            actual_lbl=actual_lbl,
            columns_lst=columns_lst,
            test_size=test_size,
            random_state=random_state)

        return

    def get_formula(self):
        lm = self.model
        col_lst = self.columns_lst

        formula = self.predicted_lbl + ' = ' + '{:.3f}'.format(lm.intercept_)
        for coef, feature in zip(lm.coef_, col_lst):
            formula += '{:+.3f}*{}'.format(coef, feature)

        return formula

    def rmse(self, prediction_s, actual_s):
        y_true = np.array(list(prediction_s.values), dtype=np.float64)
        y_pred = np.array(list(actual_s.values), dtype=np.float64)
        # print y_true
        # print "y_true.shape={};y_pred.shape={}\n".format(y_true.shape,y_pred.shape)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mae(self, prediction_s, actual_s):
        return np.sqrt(mean_absolute_error(prediction_s, actual_s))

    def rmsle(self, prediction_s, actual_s):
        """https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError"""
        n = len(prediction_s)
        pred_log = np.log(prediction_s + 1)
        actu_log = np.log(actual_s + 1)
        _sum = sum((pred_log - actu_log) ** 2)
        _rmsle = np.sqrt(_sum / n)
        return _rmsle

    def model_info(self, df):
        """

        :param df: test_df or train_df after prediction
        :return: dictionary: model_info
        """
        prediction_s = df[self.predicted_lbl]
        actual_s = df[self.actual_lbl]
        # print actual_s.head()
        # print prediction_s.head()
        model_info = {}
        model_info["model_name"] = self.model_name
        model_info["col_list"] = "&".join(self.columns_lst)
        model_info["actual_lbl"] = self.actual_lbl
        model_info["formula"] = self.get_formula()
        model_info["rmse"] = self.rmse(prediction_s=prediction_s, actual_s=actual_s)
        model_info["rmsle"] = self.rmsle(prediction_s=prediction_s, actual_s=actual_s)
        model_info["mae"] = self.mae(prediction_s=prediction_s, actual_s=actual_s)
        model_info["prediction_column"] = prediction_s,
        model_info["model"] = self.model

        # model_info["predicted_vs_actual_ax"]=  self.plot_predicted_vs_actual(df=df,actual_lbl=self.actual_lbl,predicted_lbl=self.predicted_lbl)
        return model_info
