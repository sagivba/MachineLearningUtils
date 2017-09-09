from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.ModelsUtils import ModelUtils
from MachineLearningUtils.UsefulPlots import EvaluationPlots


# load iris data into DataFrame
prd_lbl, actl_lbl = "PrdictedIrisClass", "IrisClass"
iris_df = DatasetsTools(datasets.load_iris).data_as_df(target_column_name="IrisClass")

# set clf
tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=10)

# simple usage
mu = ModelUtils(df=iris_df, model=tree_clf, predicted_lbl=prd_lbl, actual_lbl=actl_lbl)
mu.split_and_train()
results_df = mu.test_model()

# evaluate results using plot_confusion_matrix
print(mu.confusion_matrix_as_dataframe())
evp = EvaluationPlots(df=results_df, actual_lbl=actl_lbl, predicted_lbl=prd_lbl)
evp.plot_confusion_matrix(confusion_matrix=mu.confusion_matrix(), classes_lst=mu.model.classes_,
                          title="Iris-confusion_matrix")
# plt.savefig("confusion_matrix.png", bbox_inches='tight')

cr = mu.classification_report(y_pred=results_df[prd_lbl], y_true=results_df[actl_lbl])
print(cr)

evp.plot_classification_report(cr)
plt.show()
