import numpy as np
import pandas as pd

from MachineLearningUtils.UsefulMethods import UsefulMethods
from MachineLearningUtils.CommonFeatureEngineering import DataFrameManipulation


class common_titanic_things():
    def __init__(self, example_number):
        self.prd_lbl, self.actl_lbl = "PrdictedSurvived", "Survived"
        self.example_number = example_number
        self.input_path = ".\\titanic_data"
        self.output_path = ".\\titanic_data\\{}".format(example_number)
        self.map_dict = {
            "Embarked": {'S': 1, 'C': 2, 'Q': 3, np.NaN: -99},
            "Sex": {"male": 0, "female": 1}
        }
        self.age_estimator = None

    def output_csv_name(self, file_name):
        return self.output_path + "\\" + file_name

    def load_data(self, file_name):
        return pd.read_csv("{}\\{}".format(self.input_path, file_name))

    def load_train_data(self):
        return self.load_data("train.csv")

    def load_test_data(self):
        return self.load_data("test.csv")

    def load_complete_data(self):
        """
        will be use to create estimators to fillNA
        :return:
        """
        df1 = self.load_train_data()
        df2 = self.load_test_data()
        manipulator = DataFrameManipulation(df1)
        df1 = manipulator.drop_columns([self.actl_lbl])
        return pd.concat([df1, df2])

    def prepare_kaggle_file(self, mu, prep_data):
        test_df = self.load_data("test.csv")
        print("test_df:")
        print(test_df.head())

        test_df_prep = prep_data(test_df, self)

        test_df_prep.to_csv(path_or_buf=self.output_csv_name("test_data.csv"), index=False)
        print(list(test_df))
        result_df = mu.test_model(test_df_prep)

        print("results_df:")
        print(result_df.head())

        final_df = UsefulMethods.create_submition_df(test_df[["PassengerId"]], result_df[["PrdictedSurvived"]],
                                                     "PrdictedSurvived", "Survived")
        # final_df=pd.concat([test_df[["PassengerId"]],result_df[["PrdictedSurvived"]]],axis=1)

        print("final_df:")
        print(final_df.head())
        final_df.to_csv(path_or_buf=self.output_csv_name("final_df.csv"), index=False)
