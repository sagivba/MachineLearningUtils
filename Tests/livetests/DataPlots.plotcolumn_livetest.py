from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.UsefulPlots import DataPlots


def main():
    boston_dstl = DatasetsTools(datasets.load_boston)
    # print(str(boston_dstl))
    boston_df = boston_dstl.data_as_df()
    print(boston_dstl.info)

    def set_CHAS(n):
        if (n == 1):
            return "bounds river"
        else:
            None

    boston_df.CHAS = boston_df.CHAS.apply(set_CHAS)
    plotter = DataPlots(df=boston_df, ggplot=True, cmap=cm.jet)
    sizes_lst = [(4, 4), (6, 4), (6, 4), (12, 8), (7, 7), (6, 6), (6, 6), (12, 12), (12, 12), (12, 12), (12, 12),
                 (12, 12), (12, 12), (12, 12), (12, 12)]
    # plotter.plot_column(data_column=boston_df.CHAS)
    # plt.show()


    for i, col in enumerate(list(boston_df)):
        print(col)
        plotter.plot_column(data_column=boston_df[col], figsize=sizes_lst[i])
        plt.show()
    return


if __name__ == '__main__':
    main()
