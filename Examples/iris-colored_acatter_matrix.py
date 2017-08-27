from sklearn import datasets
from MachineLearningUtils.UsefulPlots import DataPlots
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
import  numpy as np


def main():

    iris = datasets.load_iris()
    _df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    plotter=DataPlots(df=_df, ggplot=True)
    fig=plotter.colored_scatter_matrix(df=_df,colored_column_name="target")
    fig.savefig("iris-colored_acatter_matrix.png")
    return

if __name__ == '__main__':
    main()