from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.UsefulPlots import DataPlots


def main():
    iris_dtst = DatasetsTools(datasets.load_iris)
    iris_df = iris_dtst.data_as_df()
    print(iris_df.info())
    plotter = DataPlots(df=iris_df, ggplot=True, cmap=cm.jet)
    sizes_lst = [(4, 4), (6, 4), (6, 4), (12, 8), (7, 7), (6, 6), (6, 6), (12, 12)]
    for i, col in enumerate(list(iris_df)):
        plotter.plot_column(data_column=iris_df[col], figsize=sizes_lst[i])
    plt.show()
    return


if __name__ == '__main__':
    main()
