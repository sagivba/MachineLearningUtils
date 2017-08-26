from sklearn import datasets
from MachineLearningUtils.UsefulPlots import DataPlots
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
import  numpy as np


def main():

    iris = datasets.load_iris()
    _df=data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    print (_df.info())
    # exit()
    plotter=DataPlots(df=_df, ggplot=True, cmap=cm.jet)
    sizes_lst=[(4,4),(6,4),(6,4),(12,8),(7,7),(6,6),(6,6),(12,12)]
    sl_axes=[]
    tr_axes=[]
    for i,col in enumerate(list(_df)):
        # ax = plotter.colored_scatter(x=_df[col], y=_df["sepal width (cm)"], z2color=_df["petal length (cm)"])
        # sl_axes.append(ax)

        ax=plotter.colored_scatter(x=_df[col], y=_df["sepal width (cm)"], z2color=_df["target"],
                                figsize=sizes_lst[i])
        tr_axes.append(ax)

    plt.show()
    return

if __name__ == '__main__':
    main()