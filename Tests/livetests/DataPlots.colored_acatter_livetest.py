from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.UsefulPlots import DataPlots


def main():
    iris_dtst = DatasetsTools(datasets.load_iris)
    iris_df = iris_dtst.data_as_df()
    print(iris_df.info())
    # exit()
    plotter = DataPlots(df=iris_df, ggplot=True, cmap=cm.jet)
    sizes_lst = [(4, 4), (6, 4), (6, 4), (12, 8), (7, 7), (6, 6), (6, 6), (12, 12)]
    tr_axes = []
    for i, col in enumerate(list(iris_df)):
        ax = plotter.colored_scatter(x=iris_df[col], y=iris_df.sepal_length_cm, z2color=iris_df.Target,
                                     figsize=sizes_lst[i])
        tr_axes.append(ax)

    plt.show()
    return


if __name__ == '__main__':
    main()
