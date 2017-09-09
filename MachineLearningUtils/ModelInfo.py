class ModelInfo():
    """
    save the model information
    this will be uset to compare between diffwrwnt models
    """

    def __init__(self,
                 model_name=None,
                 columns_lst=[],
                 prediction_column=None,
                 actual_lbl=None,
                 clf=None):
        self.model_name = model_name
        self.columns_lst = columns_lst
        self.actual_lbl = actual_lbl
        self.prediction_column = prediction_column
        self.clf = clf
        return

    def __str__(self):
        return ","
        join(self.model_name, self.actual_lbl)
        )
    def as_dict(self):
        return dict(self)


class LinearModelInfo(ModelInfo):
    def __init__(self,
                 model_name=None,
                 columns_lst=[],
                 prediction_column=None,
                 actual_lbl=None,
                 clf=None,
                 rmse=None,
                 rmsle=None,
                 mae=None,
                 formula=None,
                 ):
        ModelInfo(self,
                  model_name=model_name,
                  columns_lst=columns_lst,
                  prediction_column=prediction_column,
                  actual_lbl=actual_lbl,
                  formula=formula,
                  clf=clf, )
        self.rmse = rmse = rmse
        self.rmsle = rmsle,
        self.mae = mae,
        self.formula = formula
        pass

    def __str__(self):
        return ","
        join(self.model_name, self.actual_lbl)
