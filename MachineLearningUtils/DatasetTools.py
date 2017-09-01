import hashlib

import pandas as pd


class DatasetsTools():
    "easyier skitlearn dataset exploration mainly for unittests and demos"

    def __init__(self, load_func):
        self.load = load_func
        self.dataset = self.load()
        self._set_info()

    def _set_info(self):

        _ds = self.dataset
        _info = _ds["DESCR"]
        _info += "\n-------------------------------------\n"
        _info += "data shape  : {}\n".format(_ds.data.shape)
        _info += "target shape: {}\n".format(_ds.target.shape)
        self.info = _info
        return self.info

    def data_as_df(self, target_column_name="Target", clean_colmns_names=True):
        """
        conveert sklearn.datasets.load_X (Bunch) into pandas.DataFrame
        :param target_column_name:
        :param clean_colmns_names:
        :return:
        """
        clean_func = lambda s: str(s).translate(s.maketrans(" ,;-#$%^&*",
                                                            "__________")).replace('(', "").replace(')', '')
        _columns_names = list(self.dataset.feature_names)
        _target_column_name = target_column_name
        if clean_colmns_names:
            _columns_names = list(map(clean_func, _columns_names))
            _target_column_name = clean_func(target_column_name)
        _df = pd.DataFrame(data=self.dataset.data, columns=_columns_names)
        _df[_target_column_name] = self.dataset.target
        return _df

    def md5file(self, fname):
        """
        md5 file for unitests
        TODO(this will be moves in the futer into  another module)
        :return:
        """
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
