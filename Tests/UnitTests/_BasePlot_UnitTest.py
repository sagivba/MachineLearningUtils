import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets

from MachineLearningUtils.UsefulPlots import _BasePlot
from MachineLearningUtils.DatasetTools import DatasetsTools


class Test_BasePlot(unittest.TestCase):
    def setUp(self):
        self.iris_df = DatasetsTools(datasets.load_iris).data_as_df(target_column_name="target")
        self.boston_df = DatasetsTools(datasets.load_boston).data_as_df()
        self.fig = plt.figure()
        self.iris___BasePlot = _BasePlot(df=self.iris_df)

    def test_set_fig(self):
        created_fig = _BasePlot._set_fig()
        self.assertTrue(isinstance(created_fig, plt.Figure))
        created_fig = _BasePlot._set_fig(self.fig)
        self.assertTrue(isinstance(created_fig, plt.Figure))

    def test_set_ax(self):
        created_ax = _BasePlot._set_ax()
        ax = plt.figure().gca()
        self.assertTrue(isinstance(created_ax, plt.Axes))

        created_fig = _BasePlot._set_fig(plt.figure())
        self.assertTrue(isinstance(created_fig, plt.Figure))
        created_ax = _BasePlot._set_ax(created_fig)
        self.assertTrue(isinstance(created_ax, plt.Figure))

    def test_set_df(self):
        bp = self.iris___BasePlot
        iris_df = bp._set_df()
        expected_columns_lst = list(self.iris_df)
        actual_columns_lst = list(iris_df)
        self.assertEquals(actual_columns_lst, expected_columns_lst)

        boston_df = bp._set_df(self.boston_df)
        expected_columns_lst = list(self.boston_df)
        actual_columns_lst = list(boston_df)
        self.assertEquals(actual_columns_lst, expected_columns_lst)
