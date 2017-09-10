import pandas as pd


class ColumnManipulation():
    def __init__(self, df):
        self.df = df

    def __set_some_df(self, df, self_some_df):
        _df = df
        if not isinstance(_df, pd.DataFrame):
            _df = self_some_df
        return _df

    def _set_df(self, df=None):
        return self.__set_some_df(df, self.df)

    def map_columns_values(self, map_dict, df=None):
        _df = self._set_df(df)
        for col_name in map_dict:
            if col_name not in _df:
                print("{} not in {}".format(col_name, _df))
                continue
            dct = map_dict[col_name]
            _df[col_name] = _df[col_name].map(lambda k: dct[k])
        return _df

    def fillna_by_heuristic(self, heuristic_func, df=None):
        """
        fills na values according to the heuristic function
        :param heuristic_func:
            def heuristic_func(value,df_row=None):
                    if value: return value
                    else heuristic value by df_row ro somting like column_median,mean ...
        :param df:
        :return:
        """
        pass
