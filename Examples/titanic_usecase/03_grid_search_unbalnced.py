from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from Examples.titanic_usecase.common_titanic import common_titanic_things
from MachineLearningUtils.CommonFeatureEngineering import DataFrameManipulation
from MachineLearningUtils.ModelsUtils import ModelUtils
from MachineLearningUtils.UsefulPlots import EvaluationPlots
from MachineLearningUtils.LinearModelUtils import LinearModelUtils

"""
This is a very simple example of the usage of the tools provided by this project
the result is steel 0.71291 as it was in 01 in: https://www.kaggle.com/c/titanic
1. prepare the data:
    * replace  Embarked values and Sex values with coresponding numbers
    * drop columns ["Name","Ticket","Cabin","PassengerId"]
    * fillna most of values with median
    * fillna age using regression
    * naivly balance the data
2. create clf of DecisionTreeClassifier using gridsearch
3. use ModelUtils to split and train the data
4. predict

"""


def add_columns(df, common, manipulator):
    df["CabinsAmount"] = df.Cabin.fillna("").apply(len)
    df["SurelyAdult"] = df.Parch.apply(lambda x: int(x > 2))
    df["SurelyChild"] = df.Name.apply(lambda x: int("Master" in x))
    df["MaybeChild"] = df.Name.apply(lambda x: int("Miss" in x))
    df.loc[(df['MaybeChild'] == 1) & (df['SibSp'] == 0) & (df['Parch'] == 0), 'MaybeChild'] = 0
    df.loc[(df['MaybeChild'] == 0) & (df['SurelyChild'] == 1), 'MaybeChild'] = 1
    return df


def create_age_estimator(common):
    df = common.load_complete_data()
    manipulator = DataFrameManipulation(df)
    df = manipulator.map_columns_values(**common.map_dict)
    df = add_columns(df, common, manipulator)
    df.to_csv("create_age_estimator.csv")

    drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = manipulator.drop_columns(drop_list, df)
    col_lst = list(df)
    col_lst.remove('Age')
    for col_name in col_lst:
        df[col_name] = df[col_name].fillna(df[col_name].median())

    df_with_age = df[(df.Age.notnull())]
    # df_with_age = df_with_age[df_with_age.Age < 40]
    df_with_age = df_with_age.dropna(how='any', axis=0)
    df_no_age = df[(df.Age.isnull())]
    drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]

    lm = LinearRegression()
    mu = LinearModelUtils(df=df_with_age, lm=lm, predicted_lbl='PredAge', actual_lbl='Age')
    mu.split_and_train()
    results_df = mu.test_model()
    evp = EvaluationPlots(df=results_df, actual_lbl=mu.actual_lbl, predicted_lbl=mu.predicted_lbl)
    evp.plot_predicted_vs_actual(title="LinearRegression as fillna")
    print("LinearRegression rmse={}".format(mu.rmse(results_df.PredAge, results_df.Age)))
    plt.show()

    results_df["med_age"] = df.Age.median()
    evp = EvaluationPlots(df=results_df, actual_lbl=mu.actual_lbl, predicted_lbl="med_age")
    evp.plot_predicted_vs_actual(title="median as fillna")
    print("med_age rmse={}".format(mu.rmse(results_df["med_age"], results_df.Age)))
    plt.show()
    common.age_estimator = mu.model
    exit()


def prep_data(df, common):
    manipulator = DataFrameManipulation(df)
    df = manipulator.map_columns_values(**common.map_dict)
    df = add_columns(df, common, manipulator)
    drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = manipulator.drop_columns(drop_list, df)

    col_lst = list(df)
    col_lst.remove('Age')
    for col_name in col_lst:
        df[col_name] = df[col_name].fillna(df[col_name].median())

    def age_huristic(value, df_row=None):
        if value:
            return value
        else:
            return common.age_estimator.predict(df_row)

    # df= manipulator.fillna_by_heuristic(age_huristic,"Age",df)
    # df.Age = df.Age.astype(int)

    return df


def main():
    common = common_titanic_things(example_number='03')
    df = common.load_data("train.csv")
    create_age_estimator(common)
    # prepare the data
    df = prep_data(df, common)

    # wrire the data for later exploration
    df.to_csv(common.output_csv_name("data.csv"))
    exit()
    # create clf
    tree_clf = DecisionTreeClassifier()
    # print (tree_clf.get_params())
    param_grid = {
        # 'class_weight': None,
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6],
        'max_features': [0.2, 0.5, 0.7, 0.9, 1.0],
        # 'max_leaf_nodes': None,
        'min_impurity_decrease': [0.0, 0.1, 0.5],
        # 'min_impurity_split': None,
        'min_samples_leaf': [3, 5, 8, 10, 12],
        'min_samples_split': [10],
        'min_weight_fraction_leaf': [0.0],
        'presort': [True],
        'random_state': [123456],
        # 'splitter': 'best'
    }
    clf_gs = GridSearchCV(tree_clf, param_grid=param_grid, cv=4)
    # split and train
    mu = ModelUtils(df=df, model=clf_gs, predicted_lbl=common.prd_lbl, actual_lbl=common.actl_lbl, is_verbose=True)
    mu.is_verbose = True
    print(mu.df.head())
    mu.split_and_train()

    # test model
    train_result_df = mu.test_model()

    # evaluate tested results using plot_confusion_matrix
    print(mu.confusion_matrix_as_dataframe())
    evp = EvaluationPlots(df=train_result_df, actual_lbl=common.actl_lbl, predicted_lbl=common.prd_lbl)
    evp.plot_confusion_matrix(confusion_matrix=mu.confusion_matrix(), classes_lst=mu.model.classes_,
                              title="Titanic-confusion_matrix")
    # plt.savefig("confusion_matrix.png", bbox_inches='tight')
    cr = mu.classification_report(y_pred=train_result_df[common.prd_lbl], y_true=train_result_df[common.actl_lbl])
    print(cr)
    evp.plot_classification_report(cr)
    plt.show()
    common.prepare_kaggle_file(mu, prep_data)


if __name__ == '__main__':
    main()
