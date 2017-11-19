from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
import pandas as pd


class UsefulMethods():
    @staticmethod
    def create_submition_df(df1, df2, predicted_col_name, submition_col_name):
        """
        helpful for kaggle
        :param df1: df with columns you need
        :param df2: df with the predicted col name
        :param predicted_col_name: will rename the column name to submition_col_name
        :param submition_col_name:
        :return: df
        """
        final_df = pd.concat([df1, df2], axis=1)
        if submition_col_name not in final_df:
            final_df[submition_col_name] = final_df[predicted_col_name]
            final_df = final_df.drop(predicted_col_name, 1)
        return final_df

    @staticmethod
    def create_imputer(imputer_name, transform_func, fit_func):
        imputer_dict = {
            "transform": transform_func,
            "fit": fit_func
        }
        imputer_class = type(imputer_name, Imputer, imputer_dict)

    @staticmethod
    def make_pipeline(*steps, **kwargs):
        return make_pipeline(*steps, **kwargs)
