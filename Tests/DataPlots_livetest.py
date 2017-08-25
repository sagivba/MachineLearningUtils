from DataUtils.FileReader import *
from MachineLearningUtils.UsefulPlots import *
from sklearn import datasets


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
    if test_cmaps:
        plotter=VisPlotPlayGround(df=_df, ggplot=True, cmap=cm.jet)
        cmaps = [m for m in cm.datad if not m.endswith("_r")]
        print(sorted(cmaps))
        for c in sorted(cmaps):
            plotter.show_colormap(c)
    plt.show()
    pass

if __name__ == '__main__':
    main()