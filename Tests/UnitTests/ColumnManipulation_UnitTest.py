import unittest
from sklearn import datasets
from MachineLearningUtils.CommonFeatureEngineering import ColumnManipulation
from MachineLearningUtils.DatasetTools import DatasetsTools


class ColumnManipulationTestCase(unittest.TestCase):
    def setUp(self):
        self.boston_df = DatasetsTools(datasets.load_boston).data_as_df()
        self.iris_df = DatasetsTools(datasets.load_iris).data_as_df()

    def test_map_columns_values(self):
        iris_target = self.iris_df.Target
        count = iris_target.value_counts()
        self.assertEquals(count.sum(), 150)
        self.assertEquals(count[0], 50)
        map_dict = {
            "Target": {
                0: "Setosa",
                1: "Versicolour",
                2: "Virginica"
            }
        }

        colman = ColumnManipulation(self.iris_df)
        _df = colman.map_columns_values(map_dict)
        target_uniq = sorted(list(_df.Target.unique()))
        expected_values = sorted(map_dict["Target"].values())
        _target = _df.Target
        count = _target.value_counts()
        self.assertEquals(target_uniq, expected_values)
        self.assertEquals(count.sum(), 150)
        self.assertEquals(count.Setosa, 50)
