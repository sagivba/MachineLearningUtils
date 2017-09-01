from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.UsefulPlots import DataPlots


def main():
    boston_dtst = DatasetsTools(datasets.load_boston)
    boston_df = boston_dtst.data_as_df()
    print(boston_dtst.info)

    plotter = DataPlots(df=boston_df, ggplot=True, cmap=cm.jet)
    plotter.colored_scatter_matrix(df=boston_df, colored_column_name="Target")
    # plt.show()

    iris_dtst = DatasetsTools(datasets.load_iris)
    iris_df = iris_dtst.data_as_df()
    print(iris_df.info())
    plotter = DataPlots(df=iris_df, ggplot=True, cmap=cm.jet)
    plotter.colored_scatter_matrix(df=iris_df, colored_column_name="Target")
    plt.show()
    return


if __name__ == '__main__':
    main()
