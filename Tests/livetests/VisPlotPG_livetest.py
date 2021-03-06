import logging
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.UsefulPlots import VisPlotPlayGround


def main():
    iris = datasets.load_iris()
    _df = data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                               columns=iris['feature_names'] + ['target'])
    print(_df.info())
    # exit()
    plotter = VisPlotPlayGround(df=_df, ggplot=True, cmap=cm.jet)
    cmaps = [m for m in cm.datad if not m.endswith("_r")]
    print("\n".join(sorted(cmaps)))
    print(list(filter(lambda c: str(c).find('Rd') > -1, sorted(cmaps))))

    # for c in filter(lambda c: str(c).find('Rd')>-1, sorted(cmaps)): #[0:5]:
    for c in sorted(cmaps):  # [0:5]:
        plotter.show_colormap(c)
    plt.show()


if __name__ == '__main__':
    main()
