import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from Examples.titanic_usecase.common_titanic import common_titanic_things
from MachineLearningUtils.CommonFeatureEngineering import DataFrameManipulation
from MachineLearningUtils.ModelsUtils import ModelUtils
from MachineLearningUtils.UsefulPlots import EvaluationPlots

"""
This is a very simple example of the usage of the tools provided by this project
the result is  0.78468  in: https://www.kaggle.com/c/titanic
1. prepare the data:
    * replace  Embarked values and Sex values with coresponding numbers
    * drop columns ["Name","Ticket","Cabin","PassengerId"]
    * fillna values with median
    * naivly balance the data
2. create clf of DecisionTreeClassifier using gridsearch
3. use ModelUtils to split and train the data
4. predict

"""


def prep_data(df, common):
    col_manipulator = DataFrameManipulation(df)
    df = col_manipulator.map_columns_values(**common.map_dict)

    drop_list = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = col_manipulator.drop_columns(drop_list, df)

    for col_name in list(df):
        df[col_name] = df[col_name].fillna(df[col_name].median())

    df.Age = df.Age.astype(int)

    return df


def main():
    common = common_titanic_things(example_number='02')
    df = common.load_data("train.csv")

    # prepare the data
    df = prep_data(df, common)

    # naivly balance the data
    _df_sample = df[df.Survived == 1].sample(n=120, random_state=123456)
    df = pd.concat([df, _df_sample])

    # wrire the data for later exploration
    df.to_csv(common.output_csv_name("data.csv"))

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
