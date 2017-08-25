from DataUtils.FileReader import *
from MachineLearningUtils.UsefulPlots import *
from sklearn import datasets


def main():

    iris = datasets.load_iris()
    _df=data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    print (_df.info())
    # exit()
    plotter=VisPlotPlayGround(df=_df, ggplot=True, cmap=cm.jet)
    cmaps = [m for m in cm.datad if not m.endswith("_r")]
    print(sorted(cmaps))
    for c in sorted(cmaps):
        plotter.show_colormap(c)
    plt.show()


if __name__ == '__main__':
    main()