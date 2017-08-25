from sklearn import datasets
from MachineLearningUtils.UsefulPlots import DataPlots
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
import  numpy as np


def main():

    test_cmaps=True
    iris = datasets.load_iris()
    _df=data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    print (_df.info())
    # exit()
    plotter=DataPlots(df=_df, ggplot=True, cmap=cm.jet)
    # "sepal length,sepal width"
    plotter.colored_scatter(x=_df["sepal length (cm)"], y=_df["sepal width (cm)"], z2color=_df["petal length (cm)"])
    plotter.colored_scatter(x=_df["sepal length (cm)"], y=_df["sepal width (cm)"], z2color=_df["target"], figsize=(12, 12))
    plotter.plot_column(data_column=_df["petal width (cm)"])
    plt.show()
    return

if __name__ == '__main__':
    main()