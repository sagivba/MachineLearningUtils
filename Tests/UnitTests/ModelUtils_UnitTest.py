import unittest

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from MachineLearningUtils.DatasetTools import DatasetsTools
from MachineLearningUtils.ModelsUtils import ModelUtils


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        ds = DatasetsTools(datasets.load_iris)
        self.iris_df = ds.data_as_df(target_column_name="IrisClass")
        self.boton_df = DatasetsTools(datasets.load_boston).data_as_df()
        self.tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=10)
        self.prd_lbl = "PrdictedIrisClass"
        self.actl_lbl = "IrisClass"
        self.columns_lst = list(self.iris_df)
        self.columns_lst.pop(-1)
        self.mu = ModelUtils(df=self.iris_df, model=self.tree_clf, columns_lst=self.columns_lst,
                             predicted_lbl=self.prd_lbl, actual_lbl=self.actl_lbl)

    def test__set_df(self):
        mu = self.mu
        df = mu._set_df(None)
        iris_df = self.iris_df
        self.assertEquals(list(df), list(iris_df))
        boton_df = self.boton_df
        df = mu._set_df(boton_df)
        self.assertEquals(list(df), list(boton_df))

    def test__set_train_df(self):
        mu = self.mu
        df = mu._set_train_df(None)
        iris_df = self.iris_df
        self.assertEquals(list(df), list(iris_df))
        boton_df = DatasetsTools(datasets.load_boston).data_as_df()
        df = mu._set_train_df(boton_df)
        self.assertEquals(list(df), list(boton_df))

    def test__set_test_df(self):
        mu = self.mu
        df = mu._set_test_df(None)
        iris_df = self.iris_df
        self.assertEquals(list(df), list(iris_df))
        boton_df = DatasetsTools(datasets.load_boston).data_as_df()
        df = mu._set_test_df(boton_df)
        self.assertEquals(list(df), list(boton_df))

    def test_init(self):
        # df == None
        self.assertRaises(ValueError,
                          lambda: ModelUtils(df=None, model=self.tree_clf, columns_lst=self.columns_lst,
                                             predicted_lbl=self.prd_lbl,
                                             actual_lbl=self.actl_lbl)
                          )
        # # clf == None
        self.assertRaises(ValueError,
                          lambda: ModelUtils(df=self.iris_df, model=None, columns_lst=self.columns_lst,
                                             predicted_lbl=self.prd_lbl,
                                             actual_lbl=self.actl_lbl),
                          )
        # clf missing
        self.assertRaises(ValueError,
                          lambda: ModelUtils(df=self.iris_df, predicted_lbl=self.prd_lbl, actual_lbl=self.actl_lbl)
                          )
        mu = ModelUtils(df=self.iris_df, model=self.tree_clf, predicted_lbl=self.prd_lbl, actual_lbl=self.actl_lbl)
        self.assertIsInstance(mu, ModelUtils)

    def test_train_test_split(self):
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        # dfualt split
        train_df, test_df = mu.split_data_to_train_test()
        train_shape = train_df.shape
        self.assertTrue(train_shape[0] > 70)
        self.assertTrue(train_shape[1] == 5)

        # split another df
        train_df, test_df = mu.split_data_to_train_test(self.boton_df)
        train_shape = train_df.shape
        self.assertTrue(train_shape[0] > 350)
        self.assertTrue(train_shape[1] == 14)
        # split another test size
        train_df, test_df = mu.split_data_to_train_test(self.boton_df, test_size=0.5)
        train_shape = train_df.shape
        self.assertTrue(train_shape[0] == list(self.boton_df.shape)[0] * 0.5)
        self.assertTrue(train_shape[1] == 14)

    def test_train_model_simple(self):
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        train_df, test_df = mu.split_data_to_train_test()
        # simple train
        trained_df = mu.train_model()
        trained_shape = trained_df.shape
        self.assertEqual(trained_shape[0] * 1.0, list(self.iris_df.shape)[0] * 0.7)
        expexted_columns = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'IrisClass',
                            'PrdictedIrisClass']
        self.assertEquals(list(trained_df), expexted_columns)

    def test_train_model_diffrent_df(self):
        # train on diffrent df
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        train_df, test_df = mu.split_data_to_train_test()
        train_df1, test_df1 = mu.split_data_to_train_test(test_size=0.8)
        trained_df = mu.train_model(train_df=train_df1)
        trained_shape = trained_df.shape
        self.assertEqual(trained_shape[0], list(self.iris_df.shape)[0] * 0.2)

    def test_train_model_chosen_columns(self):
        # train on diffrent df
        mu = self.mu
        chosen_columns = ['sepal_length_cm', 'sepal_width_cm']
        self.assertIsInstance(mu, ModelUtils)
        train_df, test_df = mu.split_data_to_train_test()
        # choose columns
        trained_df = mu.train_model(columns_lst=chosen_columns)
        trained_shape = trained_df.shape  # trained shape is th orig spe+ predicted column
        self.assertEqual(trained_shape[1], len(list(self.iris_df)) + 1)

        pass

        # def test_test_model(self):
        #     pass
        #

    def test_get_X_df_and_y_s_simple(self):
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        # simple split
        train_df, test_df = mu.split_data_to_train_test()
        train_shape = train_df.shape
        self.assertEqual(train_shape[0], list(self.iris_df.shape)[0] * 0.7)

    def test_get_X_df_and_y_s_test_size(self):
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        train_df, test_df = mu.split_data_to_train_test(test_size=0.8)
        train_shape = train_df.shape
        self.assertEqual(train_shape[0], list(self.iris_df.shape)[0] * 0.2)

    def test_get_X_df_and_y_s_other_df(self):
        mu = self.mu
        self.assertIsInstance(mu, ModelUtils)
        train_df, test_df = mu.split_data_to_train_test(df=self.boton_df)
        train_shape = train_df.shape
        self.assertEqual(train_shape[0], round(list(self.boton_df.shape)[0] * 0.7))
