import unittest

from sklearn import datasets

from MachineLearningUtils.DatasetTools import DatasetsTools


class DatasetsToolsTestCase(unittest.TestCase):
    def setUp(self):
        self.boston_dtst = DatasetsTools(datasets.load_boston)
        self.iris_dtst = DatasetsTools(datasets.load_iris)

    def test_boston(self):
        self.assertIsInstance(self.boston_dtst, DatasetsTools)
        df = self.boston_dtst.data_as_df()
        self.assertEqual(list(df.shape), [506, 14])
        l = list(df)
        headrs = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'Target']
        self.assertEqual(list(df), headrs)

    def test_iris(self):
        self.assertIsInstance(self.iris_dtst, DatasetsTools)
        df = self.iris_dtst.data_as_df()
        self.assertEqual(list(df.shape), [150, 5])
        headrs = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'Target']
        self.assertEqual(list(df), headrs)
