import unittest
import pandas as pd
from sklearn import datasets
from MachineLearningUtils.CommonFeatureEngineering import DataFrameManipulation
from MachineLearningUtils.DatasetTools import DatasetsTools


class DataFrameManipulationTestCase(unittest.TestCase):
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

        colman = DataFrameManipulation(self.iris_df)
        _df = colman.map_columns_values(**map_dict)
        target_uniq = sorted(list(_df.Target.unique()))
        expected_values = sorted(map_dict["Target"].values())
        _target = _df.Target
        count = _target.value_counts()
        self.assertEquals(target_uniq, expected_values)
        self.assertEquals(count.sum(), 150)
        self.assertEquals(count.Setosa, 50)

    def test_drop_columns(self):
        drop_list = ["CRIM", "INDUS"]
        colman = DataFrameManipulation(self.boston_df)
        df = colman.drop_columns(drop_list)
        self.assertTrue("CRIM" not in list(df))
        self.assertTrue("INDUS" not in list(df))

    def test_apply_those_functions1(self):
        """apply one function on the target column and one function on the"""

        def upper_second_char(s):
            return "{}{}{}".format(s[0], str(s[1]).upper(), s[2:])

        def is_big(n):
            return n > 6

        colman = DataFrameManipulation(self.iris_df)
        _df = colman.map_columns_values(Target={0: "Setosa", 1: "Versicolour", 2: "Virginica"})
        self.assertTrue("Setosa" in _df.Target.unique())

        _df = colman.apply_those_functions(Target=upper_second_char, sepal_length_cm=is_big)
        self.assertTrue("SEtosa" in _df.Target.unique())

        self.assertTrue(set([True, False]) == set(_df.sepal_length_cm.unique()))

    def test_apply_those_functions2(self):
        def upper_second_char(s):
            return "{}{}{}".format(s[0], str(s[1]).upper(), s[2:])

        def concat_len(s):
            return str(s) + str(len(s))

        func_list = [upper_second_char, concat_len]
        colman = DataFrameManipulation(self.iris_df)
        _df = colman.map_columns_values(Target={0: "Setosa", 1: "Versicolour", 2: "Virginica"})
        self.assertTrue("Setosa" in _df.Target.unique())

        _df = colman.apply_those_functions(Target=func_list)
        self.assertTrue("SEtosa6" in _df.Target.unique())

    def test_split_columns_into_columns(self):
        def under_split(s): return str(s).split('_')

        colman = DataFrameManipulation(self.iris_df)
        _df = colman.map_columns_values(Target={0: "s_e_t_o_s_a", 1: "v_e_r_s_i_colour", 2: "V_i_rginica"})
        self.assertTrue("s_e_t_o_s_a" in _df.Target.unique())

        col_names = [c for c in "abcdefghyjk"]
        splited_df = colman.split_columns_into_columns(col_to_split="Target", new_columns_names_lst=col_names,
                                                       split_func=under_split)
        self.assertEquals(_df.shape, _df.shape)
        self.assertEquals(_df.shape, splited_df.shape)
        b_lst = splited_df[splited_df.Target == 's_e_t_o_s_a'].b.unique()
        f_lst = splited_df[splited_df.Target == 'v_e_r_s_i_colour'].f.unique()

        self.assertTrue(b_lst[0] == 'e')
        self.assertTrue(f_lst[0] == 'colour')

        self.assertTrue(set([True, False]) == set(_df.sepal_length_cm.unique()))
