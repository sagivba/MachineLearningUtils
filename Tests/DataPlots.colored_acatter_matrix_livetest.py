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
    print (_df.info())
    # exit()
    plotter=DataPlots(df=_df, ggplot=True, cmap=cm.jet)
    plotter.colored_scatter_matrix(df=_df,colored_column_name="target")

    # diabetes=datasets.load_diabetes()
    # _df= pd.DataFrame(data= np.c_[diabetes['data'], diabetes['target']],
    #                  columns= diabetes['feature_names'] + ['target'])
    # print (_df.info())
    # plotter = DataPlots(df=None, ggplot=True, cmap=cm.jet)
    # plotter.colored_scatter_matrix(df=_df, colored_column_name="target")
    plt.show()
    return

if __name__ == '__main__':
    main()