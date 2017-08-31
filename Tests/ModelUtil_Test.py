import unittest
from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.ModelsUtils import ModelUtils
from MachineLearningUtils.UsefulPlots import EvaluationPlots
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib import pyplot as plt
from sklearn import datasets

# set the data
prd_lbl, actl_lbl = "PrdictedIrisClass", "IrisClass"
iris_df = DatasetsTools(datasets.load_iris).data_as_df(target_column_name="IrisClass")
# set clf
tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=10)

# simple usage
mu = ModelUtils(df=iris_df, clf=tree_clf, predicted_lbl=prd_lbl, actual_lbl=actl_lbl)
# print (iris_df.head())
train_df, test_df = mu.train_test_split()
mu.train_model()
test_result_df = mu.test_model()
# print (test_result_df.head(20))

# evaluate results
print(mu.confusion_matrix_as_dataframe())
evp = EvaluationPlots(df=test_result_df, actual_lbl=mu.actual_lbl, predicted_lbl=mu.predicted_lbl)
evp.plot_confusion_matrix(confusion_matrix=mu.confusion_matrix(), classes_lst=mu.clf.classes_)
plt.savefig("cm.png", bbox_inches='tight')
# plt.show()
