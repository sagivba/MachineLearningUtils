import pandas as pd
import collections
import inspect


class DataFrameManipulation():
    """
    drop list of columns, fillna using smarter functions, map columns values in one line and more goodies..
    """
    def __init__(self, df):
        self.df = df

    def __set_some_df(self, df, self_some_df):
        _df = df
        if not isinstance(_df, pd.DataFrame):
            _df = self_some_df
        return _df

    def _set_df(self, df=None):
        return self.__set_some_df(df, self.df)

    def split_columns_into_columns(self, col_to_split, new_columns_names_lst: list, split_func, df=None):
        # TODO
        """

        :param col_to_split: Series
        :param new_columns_names_lst: list of the new columns names
        :param split_func: with one param for example: lambda s:str(s).split(',')
        :param df:
        :return:
        """
        _df = self._set_df(df)

        data = list(_df[col_to_split].apply(split_func))
        new_df = pd.DataFrame(data, index=_df.index)
        for i, col_name in enumerate(list(new_df)):
            new_df = self.rename_column(col_name, new_columns_names_lst[i], df=new_df)

        _df = pd.concat([_df, new_df], axis=1, verify_integrity=True)
        return _df

    def map_columns_values(self, df=None, **kwargs):
        _df = self._set_df(df)
        for col_name, dict_item in kwargs.items():
            # for col_name in map_dict:
            if col_name not in _df:
                print("{} not in {}".format(col_name, _df))
                continue
            _df[col_name] = _df[col_name].map(lambda k: dict_item[k])
            self.df = _df
        return _df

    def drop_columns(self, columns_list, df=None):
        _df = self._set_df(df)
        _df = _df.drop(columns_list, 1)
        self.df = _df
        return _df

    def rename_column(self, old_name, new_name, df=None):
        _df = self._set_df(df)
        _df = _df.rename(index=str, columns={old_name: new_name})
        return _df

    def fillna_by_heuristic(self, heuristic_func, col_name, df=None):
        """
        fills na values according to the heuristic function
        :param heuristic_func:
            def heuristic_func(value,df_row=None):
                    if value: return value
                    else heuristic value by df_row ro somting like column_median,mean ...
        :param df:
        :return:
        """
        _df = self._set_df(df)
        if col_name not in list(_df):
            raise ValueError("{} not in columns list".format(col_name))

        if str(inspect.signature(heuristic_func)) == '(value)':
            df[col_name] = df[col_name].apply(heuristic_func)
        elif str(inspect.signature(heuristic_func)) == '(value, df_row=None)':
            df[col_name] = df.apply(heuristic_func, axis=1)
        else:
            raise ValueError("heuristic_func signature:{} but should be '(value, df_row=None)' ".format(
                nspect.signature(heuristic_func)))

        return _df

    def apply_those_functions(self, df=None, **kwargs):
        _df = self._set_df(df)
        for col_name in kwargs:
            func_list = kwargs[col_name]
            if col_name not in list(_df):
                raise ValueError("{} not in columns list".format(col_name))

            if callable(func_list):
                func_list = [func_list]
            if not isinstance(func_list, collections.Iterable):
                raise ValueError("{} is not iterable (is should be list of functions)".format(func_list))
            for func in func_list:
                _df[col_name] = _df[col_name].apply(func)
        return _df
