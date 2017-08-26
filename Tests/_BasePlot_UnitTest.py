import unittest

from sklearn import datasets
from MachineLearningUtils.UsefulPlots import _BasePlot
from matplotlib import cm
from matplotlib import pyplot as plt
import pandas as pd
import  numpy as np

class TestBasPlot(unittest.TestCase):
    def setUp(self):
        self.iris = datasets.load_iris()
        self.df = pd.DataFrame(data=np.c_[self.iris['data'], self.iris['target']],
                                   columns=self.iris['feature_names'] + ['target'])
        self.fig=plt.figure()

    def test_set_fig(self):
        created_fig=_BasePlot._set_fig()
        self.assertTrue(isinstance(created_fig,plt.Figure))
        created_fig = _BasePlot._set_fig(self.fig)
        self.assertTrue(isinstance(created_fig,plt.Figure))

    def test_set_ax(self):
        created_ax = _BasePlot._set_ax()
        ax=plt.figure().gca()
        self.assertTrue(isinstance(created_ax,plt.Axes))

        created_fig = _BasePlot._set_fig(plt.figure())
        self.assertTrue(isinstance(created_fig, plt.Figure))
        created_ax = _BasePlot._set_ax(created_fig)
        self.assertTrue(isinstance(created_ax,plt.Figure))



