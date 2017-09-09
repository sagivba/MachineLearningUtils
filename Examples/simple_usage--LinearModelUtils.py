from matplotlib import pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.LinearModelUtils import LinearModelUtils
from MachineLearningUtils.UsefulPlots import EvaluationPlots

pd.set_option('expand_frame_repr', False)

# load boston data into DataFrame
prd_lbl, target = "PrdictedPrice", "Price"
boston_dtst = DatasetsTools(datasets.load_boston).data_as_df(target_column_name=target)
print(boston_dtst.describe())
boston_df = boston_dtst.data_as_df(target_column_name=target)
print(boston_df.head())
# set linear model
lm = LinearRegression()

# simple usage
mu = LinearModelUtils(df=boston_df, lm=lm, predicted_lbl=prd_lbl, actual_lbl=target)
mu.split_and_train()
results_df = mu.test_model()
# evaluate results using plot_confusion_matrix
print(mu.get_formula())
evp = EvaluationPlots(df=results_df, actual_lbl=mu.actual_lbl, predicted_lbl=mu.predicted_lbl)
evp.plot_predicted_vs_actual()
# plt.savefig("boston-plot_predicted_vs_actual.png", bbox_inches='tight')
plt.show()
