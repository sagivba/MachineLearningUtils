import pandas as pd
import sklearn
from  functools import partial

if sklearn.__version__ < "18.0":
    from sklearn.cross_validation import train_test_split as trn_tst_split
else:
    from sklearn.model_selection import train_test_split as trn_tst_split

__version__ = "0.0.2"


class ModelUtils():
    """
    utils for easier skitlearn classifier handling
    """

    @classmethod
    def create_model_name(cls, model):
        return str(type(model)).split('.')[-1].split("'")[0]

    def verbose(self, msg, caller="", severity="INFO"):
        if self.is_verbose:
            print("{:>6} {:<15}: {}".format(severity, caller, msg))

    def __str__(self):
        lst = list(self.__dict__.items())
        lst = list(filter(lambda x: not hasattr(x[1], '__iter__'), lst))
        return str(lst)

    def __init__(self,
                 df,
                 model=None,
                 model_name="",
                 predicted_lbl=None,
                 actual_lbl=None,
                 columns_lst=[],
                 test_size=0.3,
                 random_state=123456,
                 is_verbose=False
                 ):
        """

        :param df: pandas Dataframe
        :param model:
        :param model_name: name for the clf object
        :param predicted_lbl:  name of the column with the predected values
        :param actual_lbl: name of the column with the actual  values (prediced__lbl is tested against)
        :param columns_lst: subset of the df columns names that clf will fit by them
        :param test_size: [0:1.0]
        :param random_state:
        :param is_verbose: is verbodse mode (mainly for debugging)
        """
        pd.set_option('expand_frame_repr', False)
        self.is_verbose = is_verbose
        self.df = df
        if not model:
            raise ValueError("missing clf")
        self.model = model
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = "{}==>{}".format(self.create_model_name(model), actual_lbl)

        if columns_lst:
            self.columns_lst = columns_lst
        else:
            self.columns_lst = list(df)
            self.columns_lst.remove(actual_lbl)
        self.predicted_lbl = predicted_lbl
        self.actual_lbl = actual_lbl
        self.test_size = test_size
        self.random_state = random_state
        if (predicted_lbl or actual_lbl) and self.df is None:
            raise ValueError("predicted_lbl or actual_lbl are defined but df is None")
        if predicted_lbl in self.columns_lst:
            raise ValueError("predicted_lbl can not be one of columns in the columns list")
        if actual_lbl in self.columns_lst:
            raise ValueError("actual_lbl can not be one of columns in the columns list")
        self.validate_col_lst(df, self.columns_lst)
        self.verbose(msg="{}".format(self), caller=self.__class__.__name__)
        self.train_df, self.test_df = None, None
        self.split_data_to_train_test()  # this wiil initialze self.train_df, self.test_df

        return

    def __set_something(self, thing, self_thing, caller=None, expeted_type=None):
        _thing = thing
        if _thing is None or not _thing:
            _thing = self_thing
        if type is not None and type(_thing) != expeted_type:
            raise TypeError("{}: type of {} is not {}".format(caller, _thing, expeted_type))
        return _thing


    def __set_some_df(self, df, self_some_df):
        _df = df
        if not isinstance(_df, pd.DataFrame):
            _df = self_some_df
        return _df

    def _set_some_df(self, self_some_df, df):
        _df = df
        if not isinstance(_df, pd.DataFrame):
            _df = self_some_df
        return _df

    def _set_df(self, df=None):
        return self.__set_some_df(df, self.df)

    def _set_train_df(self, train_df=None):
        return self.__set_some_df(train_df, self.train_df)

    def _set_test_df(self, test_df=None):
        return self.__set_some_df(test_df, self.test_df)

    def _set_tested_df(self, tested_df):
        return self.__set_some_df(tested_df, self.test_df)  # after selt.test_model self.test_df is tested_df

    def _set_test_size(self, test_size):
        return self.__set_something(test_size, self.test_size, self._set_test_size.__name__, float)

    def _set_column_lst(self, columns_lst):
        return self.__set_something(columns_lst, self.columns_lst, self._set_column_lst.__name__, list)

    def _set_actual_lbl(self, actual_lbl):
        return self.__set_something(actual_lbl, self.actual_lbl, self._set_actual_lbl.__name__, str)

    def _set_predicted_lbl(self, predicted_lbl):
        return self.__set_something(predicted_lbl, self.predicted_lbl, self._set_predicted_lbl.__name__, str)

    def _set_cm(self, cm):
        return self.__set_something(cm, self.cm, self._set_cm.__name__)

    def validate_col_lst(self, df, columns_lst):
        """
        check that columns_lst is tbset of self.df.columns.names
        :param df:
        :param columns_lst:
        :return:
        """
        if columns_lst == []:
            raise ValueError("column_lst is empty")
        col_set = set(columns_lst)
        df_col_set = set(list(df))
        if col_set - df_col_set != set():
            raise ValueError("col_lst has columns name that does not exists in the DataFrame columns")
        return True

    def split_data_to_train_test(self, df=None, test_size=None):
        """

        :param df: defualt is self df but you can also give it as parameter
        :param test_size: 0:1.0
        :return:
        """
        func_name = self.split_and_train.__name__

        _df = self._set_df(df)
        _test_size = self._set_test_size(test_size)

        self.train_df, self.test_df = trn_tst_split(
            _df,
            test_size=_test_size,
            random_state=self.random_state
        )
        msg = "test_size= {}, train_df shape: {}, test_df shape: {}".format(_test_size, self.train_df.shape,
                                                                            self.test_df.shape)
        self.verbose(msg=msg, caller=func_name)
        return self.train_df, self.test_df

    def get_X_df(self, df):
        return df[self._x_lbl]

    def get_X_df_and_y_s(self, df=None, columns_lst=None, actual_lbl=None):
        _df = df
        if not isinstance(df, pd.DataFrame):
            _df = self.df

        _columns_lst = self._set_column_lst(columns_lst)
        self.validate_col_lst(_df, _columns_lst)
        _actual_lbl = self._set_actual_lbl(actual_lbl)

        X = _df.ix[:, _columns_lst]
        y = _df[_actual_lbl]

        return X, y

    def train_model(self, train_df=None, columns_lst=None, actual_lbl=None):
        _df = self._set_train_df(train_df)
        _columns_lst = self._set_column_lst(columns_lst)
        _actual_lbl = self._set_actual_lbl(actual_lbl)

        if self.predicted_lbl in list(_df):
            _df.drop(self.predicted_lbl, axis=1)

        X_df, y_s = self.get_X_df_and_y_s(df=_df, columns_lst=_columns_lst, actual_lbl=_actual_lbl)
        self._x_lbl = list(X_df)
        # print list(X_df)
        self.model.fit(X_df, y_s)

        pred_df = pd.DataFrame(data={self.predicted_lbl: self.model.predict(X_df)}, index=_df.index)
        _df = pd.concat([_df, pred_df], axis=1, join_axes=[_df.index])
        # self._set_train_df(_df)
        # print "{} score is: {}\n".format(self.clf_name, self.clf.score(X_df, y_s))

        return _df

    def test_model(self, test_df=None, columns_lst=None):
        _df = test_df
        if not isinstance(_df, pd.DataFrame):
            _df = self.test_df
        if self.predicted_lbl in list(_df):
            _df.drop(self.predicted_lbl, axis=1, inplace=True)

        X_df, y_s = self.get_X_df_and_y_s(_df, columns_lst)

        #  of course we tont fit test...
        pred_df = pd.DataFrame(data={self.predicted_lbl: self.model.predict(X_df)}, index=_df.index)

        _df = pd.concat([_df, pred_df], axis=1, join_axes=[_df.index])
        # print np.log10(_df[self.predicted_lbl]+1)
        if not test_df:
            self.test_df = _df
        return _df

    def split_and_train(self):
        train_df, test_df = self.split_data_to_train_test()
        self.train_model()
        return self.train_df, self.test_df

    def confusion_matrix(self, tested_df=None, actual_lbl=None, predicted_lbl=None):
        _tested_df = self._set_tested_df(tested_df)
        _actual_lbl = self._set_actual_lbl(actual_lbl)
        columns_lst = list(_tested_df)
        if not _actual_lbl in columns_lst:
            raise KeyError("actual_lbl={} is not in the columns of in tested_df:{}".format(_actual_lbl, columns_lst))
        _predicted_lbl = self._set_predicted_lbl(predicted_lbl)
        if not _predicted_lbl in columns_lst:
            raise KeyError(
                "predicted_lbl{} is not in the columns of in tested_df:{}".format(_predicted_lbl, columns_lst))

        self.cm = sklearn.metrics.confusion_matrix(y_true=_tested_df[_actual_lbl], y_pred=_tested_df[_predicted_lbl])
        return self.cm

    def confusion_matrix_as_dataframe(self, tested_df=None, actual_lbl=None, predicted_lbl=None):
        _tested_df = self._set_tested_df(tested_df)
        _actulal_lbl = self._set_actual_lbl(actual_lbl)
        _predicted_lbl = self._set_predicted_lbl(predicted_lbl)
        _confusion_matrix = self.confusion_matrix(_tested_df, _actulal_lbl, _predicted_lbl)

        return pd.DataFrame(
            _confusion_matrix,
            index=self.model.classes_,
            columns=self.model.classes_)
