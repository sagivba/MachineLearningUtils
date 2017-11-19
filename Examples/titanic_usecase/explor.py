from MachineLearningUtils.ModelsUtils import ModelUtils
from MachineLearningUtils.CommonFeatureEngineering import DataFrameManipulation
from MachineLearningUtils.UsefulPlots import *
import pandas as pd

input_path = ".\\titanic_data"
output_path = ".\\titanic_data\\_explore"
pd.set_option('expand_frame_repr', False)


def prep_data(df):
    map_dict = {
        "Embarked": {'S': 'S', 'C': 'C', 'Q': 'Q', np.NaN: 'Nan_str'},

        "Survived": {0: 'No', 1: 'Yes', 3: "Unknown"}
    }
    manipulator = DataFrameManipulation(df)
    df = manipulator.map_columns_values(**map_dict)
    age_mid = df.Age.mean()
    for col_name in list(df):
        df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    def fillna_age(age):
        if age not in [None, np.NaN]:
            return age
        return age_mid

    def fix_age(age):
        age = int(age)
        if age < 2:
            return 1
        return age

    def number_of_cabins(c):
        if c in [None, np.NaN]: return 0
        return len(c.split())

    df["number_of_cabins"] = df.Cabin.apply(number_of_cabins)

    df = manipulator.apply_those_functions(Age=fix_age, Fare=int)
    for col_name in list(df):
        df[col_name] = df[col_name].fillna(df[col_name].mode()[0])
    df.Age = df.Age.astype(int)

    return df


train_df = pd.read_csv(".\\titanic_data\\train.csv")
test_df = pd.read_csv("{}\\test.csv".format(input_path))
test_df["Survived"] = 3
df = pd.concat([train_df, test_df])
df = prep_data(df)
print(df.info)
print(df.head())
df.to_csv(path_or_buf="{}\\data.csv".format(output_path))


def heuristic_func(value):
    pass


import inspect

print(inspect.signature(heuristic_func))
# dtp=DataPlots(df=train_df,is_verbose=True)
# # print (train_df.info())
# # print (train_df.head())
# # print (list(train_df))
# # d=train_df.Embarked.value_counts()
# # print (d)
# # exit()
# map_dict={
#     "Pclass":{1 : "first",2:"second",3:"third"}
# }
# train_df=DataFrameManipulation(train_df).map_columns_values(map_dict)
#
#
# # columns_to_plot=set(list(train_df))-set(('PassengerId', 'Survived'))
# # for col_name in sorted(columns_to_plot):
# #     dtp.plot_column(train_df[col_name])
# #     plt.show()
# plotter = DataPlots(df=train_df, ggplot=True,is_verbose=True)
# _df = train_df.copy()
# print (list(_df))
# _z2color=_df.Survived
# _z2color = _z2color[(_df["Fare"].notnull()) & (_df["Parch"].notnull())]
# _df = _df[(_df["Fare"].notnull()) & (_df["Parch"].notnull())]
#
# fig = plotter.colored_scatter_matrix(df=train_df, colored_column_name="Survived")
# #fig= plotter.colored_scatter(_df.Fare,_df.Parch,_z2color)
# plt.show()
# exit()
# # fig = plotter.colored_scatter_matrix(df=train_df, colored_column_name="Survived")
