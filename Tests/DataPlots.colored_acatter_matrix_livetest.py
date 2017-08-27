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

    plotter=DataPlots(df=_df, ggplot=True, cmap=cm.jet)
    plotter.colored_scatter_matrix(df=_df,colored_column_name="target")
    # plt.show()

    boston=datasets.load_boston()
    _columns=list(boston['feature_names'])
    _columns.append('target')
    _data=np.c_[boston['data'], boston['target']]
    _df= pd.DataFrame(data=_data , columns= _columns)
    print (_df.info())
    plotter = DataPlots(df=_df, ggplot=True, cmap=cm.jet)
    plotter.colored_scatter_matrix(df=_df, colored_column_name="target")
    plt.show()
    return

if __name__ == '__main__':
    main()