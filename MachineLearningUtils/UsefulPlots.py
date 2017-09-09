import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import sklearn

# if sklearn.__version__ < "18.0":
#     from sklearn.cross_validation import validation_curve
# else:
#     from sklearn.model_selection import validation_curve

__version__ = '0.0.5'


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


class _BasePlot():
    """
    collection of class methods and more for the plots class below
    """

    def verbose(self, msg, caller="", severity="INFO"):
        if self.is_verbose:
            print("{:>6} {:<15}: {}".format(severity, caller, msg))

    def __init__(self, df=None, ggplot=True, cmap=cm.OrRd, is_verbose=False):
        self.is_verbose = is_verbose
        self.df = df
        self.ggplot = ggplot
        if ggplot:
            self.use_ggplt()
        self.cmap = cmap

    def use_ggplt(self):
        caller = self.use_ggplt.__name__
        self.verbose("use ggplot", caller)
        plt.style.use('ggplot')

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

    def _set_df(self, df=None):
        return self.__set_some_df(df, self.df)


    @classmethod
    def _set_ax(cls, ax=None):
        if ax:
            _ax = ax
        else:
            _fig = plt.figure()
            _ax = _fig.gca()
        return _ax

    @classmethod
    def _set_fig(cls, fig=None, figsize=None):
        _figsize = cls._set_figsize(figsize)
        if fig is not None:
            _fig = fig
        else:
            _fig = plt.figure(figsize=_figsize)
        return _fig

    @classmethod
    def _set_subplot_info(cls, subplot_info=None):
        if subplot_info is not None:
            _subplot_info = subplot_info
        else:
            _subplot_info = (1, 1, 1)
        return _subplot_info

    @classmethod
    def _set_figsize(cls, figsize=None):
        if figsize is not None:
            _figsize = figsize
        else:
            _figsize = (8, 8)
        return _figsize

    @classmethod
    def _set_title(cls, title=None, x_name='x', y_name='y'):
        if title is not None:
            _title = title
        else:
            _title = "{} vs. {}".format(x_name, y_name)
        return _title

    @classmethod
    def _set_color_func(cls, color_func=None):
        ### --- start defualt_color_func --- ###
        def defualt_color_func(z):
            """
            this is the defualt color_func
            :param z:
            :return: color_func
            """
            numeric_dtypes = ['int32', 'int64', 'float32', 'float64']

            if z.dtype.name in numeric_dtypes:
                _z_col = z
            else:
                def to_str(s):
                    if s:
                        return str(s)
                    if np.isnan(s):
                        return "--np.Nan--"
                    else:
                        return "--None--"

                uniq_list = z.unique()
                # print(uniq_list)
                sorted_uniq = sorted(map(to_str, uniq_list))
                map_dict = dict([(k, i) for i, k in enumerate(sorted_uniq)])
                _z_col = z.map(lambda x: map_dict[x])
            scaler = MinMaxScaler(copy=True, feature_range=(30, 120))
            _Z = _z_col.to_frame()
            scaler.fit(_Z)
            _Z = scaler.transform(_Z)
            return _Z[:, 0]

        ### --- end defualt_color_func --- ###

        if color_func is not None:
            _color_func = color_func
        else:
            _color_func = defualt_color_func

        return _color_func


class VisPlotPlayGround(_BasePlot):
    """
    playground for visualization (color map and more...
    """

    def show_colormap(self, cmap, subplot_info=None):
        """
        show the gradiant of cmap
        :param cmap:
        :param subplot_info:
        :return:
        """
        # self.verbose(self.show_colormap.__name__)
        im = np.outer(np.ones(10), np.arange(100))
        _fig, axes = plt.subplots(2, figsize=(6, 1.5),
                                  subplot_kw=dict(xticks=[], yticks=[]))
        _fig.subplots_adjust(hspace=0.1)
        axes[0].set_title("{}".format(cmap), fontsize=14)
        axes[0].imshow(im, cmap=cmap)
        axes[1].imshow(im, cmap=self.grayify_cmap(cmap))

        return _fig, axes

    def grayify_cmap(self, cmap):
        """
        Return a grayscale version of the colormap
        :param cmap:
        :return:
        """
        self.verbose("{} - {}".format(self.show_colormap.__name__, cmap))
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived greyscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


class DataPlots(_BasePlot):
    """
    here  are some very useful plot for data exploration:
    colored_scatter plot scatter of x vs y whth color of third element

    """

    def colored_scatter(self, x, y, z2color, s=50, ax=None, figsize=None, color_func=None, title=None):
        """
        plot scatter of x vs y whth color of third element
        :param x: pandas series object
        :param y: pandas series object
        :param z2color: pandas series object
        :param ax    : plt.Axes
        :param figsize: (lengrh,witdh) inches
        :param color_func: input pandas series object - return pandas series object  of colors
        :return:
        """
        func_name = self.colored_scatter.__name__
        _figsize = figsize
        if figsize is not None and ax is not None:
            raise ValueError("both ax and figsize were  given - only one should be used.")
        elif figsize is not None:
            _figsize = self._set_figsize(figsize=figsize)
            _fig = self._set_fig(fig=None, figsize=_figsize)
            _ax = self._set_ax(_fig.gca())
        else:

            _ax = self._set_ax(ax)

        _color_func = self._set_color_func(color_func)
        if title is not None:
            _title = self._set_title(title=title)
            _ax.set_title(_title, fontsize=14)
        _ax.set_xlabel(x.name, fontsize=12)
        _ax.set_ylabel(y.name, fontsize=12)
        _ax.grid(True, linestyle='-', color='0.75')
        # scatter with colormap mapping to z value
        # Nones replaced with stirng
        _z2color = z2color.fillna("--None--")
        # print("_z2color={}".format(_z2color.unique()))
        # print("x={}".format(x.unique()))
        # print("y={}".format(y.unique()))
        colors = _color_func(_z2color)
        # print( pd.Series(colors).unique())
        _ax.scatter(x, y, s=s, c=colors, marker='o', cmap=self.cmap);

        self.verbose("s={}, x_lable = {:15}, y_lable = {:15}, title= {:15}".format(s, _ax.xaxis.label._text,
                                                                                   _ax.yaxis.label._text,
                                                                                   _ax.title._text), func_name)
        return _ax

    def plot_column(self, data_column, fig=None, figsize=None):
        func_name = self.colored_scatter.__name__

        plt.style.use('ggplot')
        plot_dict = {
            "object": [{"kind": "bar"}],
            "bool": [{"kind": "bar"}],
            "float64": [{"kind": "box"}, {"kind": "line"}, {"kind": "hist"}],
            "int64": [{"kind": "box"}, {"kind": "hist"}],
            "float32": [{"kind": "box"}, {"kind": "line"}],
            "int32": [{"kind": "box"}, {"kind": "hist"}],
        }
        _data = data_column
        # fig_path = self.config.fig_path("{}.png".format(_data.name))
        _fig = self._set_fig(fig)
        axes = []
        plot_lst = plot_dict[str(_data.dtype)]

        for i, plitm in enumerate(plot_lst):
            axes.append(_fig.add_subplot(len(plot_lst), 1, i + 1))
            _kind = plitm['kind']
            if str(_data.dtype) in ['object', 'int64', 'bool'] and len(_data.unique()) < 50:
                _df = _data.apply(pd.value_counts)
                x_val = list(_df)
                y_val = _df.sum()
                ax_title = "{}-{}-{}".format(_data.name, _kind, 'count')
                axes[i].set_title(ax_title)
                _ylim = [0, 1.1 * max(y_val)]
                axes[i].set_ylim(_ylim)
                self.verbose("axes[{}] title={} ylim={}".format(i, ax_title, _ylim))
                y_val.plot(kind='bar', title=_data.name + ' count', ax=axes[i])
            elif str(_data.dtype) not in ['object', 'bool']:
                if (_kind == 'box'):
                    _min = min(0.9 * min(data_column), 1.1 * min(data_column), -0.1)
                    _max = max(0.9 * max(data_column), 1.1 * max(data_column), 1.1)
                    _ylim = (_min, _max)
                    self.verbose("kind={}, ylim={}".format(_kind, _ylim), func_name)
                    axes[i].set_ylim(_ylim)
                _data.plot(kind=_kind, ax=axes[i])
            else:
                return None
        return _fig

    def colored_scatter_matrix(self, df, colored_column_name, figsize=(12, 12), **kwargs):
        """
        Plots a scatterplot matrix of subplots.
        Each row of "data" is plotted against other rows, resulting in a
        nrows by nrows grid of subplots with the diagonal subplots labeled with "names".
        Additional keyword arguments are passed on to matplotlib's "plot" command.
        Returns the matplotlib figure
        object containg the subplot grid.
        """
        func_name = self.colored_scatter_matrix.__name__
        if not colored_column_name:
            raise ValueError("colored_column_name is missing")
        if not colored_column_name in list(df):
            raise ValueError("{} is not on of df columns: {}".format(colored_column_name, list(df)))

        _figsize = self._set_figsize(figsize)
        _df = self._set_df(df)
        _z2color = _df[colored_column_name]
        _df = _df.drop(colored_column_name, axis=1)


        # plot only numeric type
        numeric_dtypes = ('int64', 'float64', 'int32', 'float32')
        for i, col_x in enumerate(_df):
            if str(_df[col_x].dtype) not in numeric_dtypes:
                _df = _df.drop(col_x, axis=1)

        _nrecords, _ncols = _df.shape
        self.verbose(
            "number of numeric columns: {}, number of records: {}, figsize={}".format(_ncols, _nrecords, _figsize),
            func_name)

        _fig, axes = plt.subplots(nrows=_ncols, ncols=_ncols, figsize=_figsize)
        _fig.subplots_adjust(hspace=0.05, wspace=0.05)
        # print(_df.head())
        # Hide all ticks and labels
        for ax in axes.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        _df_copy = _df.copy()
        # Plot the data.
        for i, col_x in enumerate(_df):
            for j, col_y in enumerate(_df):
                # print("{},{}==>{},{}".format(i, j, col_x, col_y))
                _df = _df_copy.copy()
                # _df["_z2color"] = _z2color
                _z2color = _z2color[(_df[col_x].notnull()) & (_df[col_y].notnull())]
                _df = _df[(_df[col_x].notnull()) & (_df[col_y].notnull())]

                axes[i, j] = self.colored_scatter(x=_df[col_x], y=_df[col_y], z2color=_z2color, s=20, ax=axes[i, j])

        # Label the diagonal subplots...
        for i, label in enumerate(list(_df)):
            axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center')

        # Turn on the proper x or y axes ticks.
        for i, j in zip(range(_ncols), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            axes[i, j].yaxis.set_visible(True)

        return _fig


class EvaluationPlots(_BasePlot):
    def __init__(self,
                 df,
                 actual_lbl,
                 predicted_lbl,
                 title=None,
                 linear_model_name="",
                 ggplot=True,
                 cmap=cm.OrRd,
                 is_verbose=False
                 ):
        _BasePlot.__init__(self, df, ggplot, cmap, is_verbose)
        self.model_name = linear_model_name
        self.actual_lbl = actual_lbl
        self.predicted_lbl = predicted_lbl
        if title:
            self.title = title
        else:
            self.title = "{}={}".format(linear_model_name, predicted_lbl)
        self.verbose(msg="{}".format(self), caller=self.__class__.__name__)

    def __str__(self):
        lst = list(self.__dict__.items())
        lst = list(filter(lambda x: not hasattr(x[1], '__iter__'), lst))
        return str(lst)

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

    def _set_actual_lbl(self, actual_lbl):
        return self.__set_something(actual_lbl, self.actual_lbl, self._set_actual_lbl.__name__, str)

    def _set_predicted_lbl(self, predicted_lbl):
        return self.__set_something(predicted_lbl, self.predicted_lbl, self._set_predicted_lbl.__name__, str)

    def plot_predicted_vs_actual(self,
                                 df=None,
                                 dot_size=10,
                                 xlim=None,
                                 ylim=None,
                                 title=None,
                                 actual_lbl=None,
                                 predicted_lbl=None
                                 ):
        """
        This method creates sctter plot of predicted values vs the actual valus.

        :param df:
        :param dot_size:
        :param xlim:
        :param ylim:
        :param title:
        :param actual_lbl:
        :param predicted_lbl:
        :return:
        """
        func_name = self.plot_predicted_vs_actual.__name__
        _df = self._set_df(df)
        _title = self._set_title(title)
        _actual_lbl = self._set_actual_lbl(actual_lbl)
        _predicted_lbl = self._set_predicted_lbl(predicted_lbl)
        self.verbose("title={}, actual_lbl={},predicted_lbl={} ".format(_title, _actual_lbl, _predicted_lbl), func_name)
        if not xlim:
            xlim = {"min": _df[_actual_lbl].min(), "max": _df[_actual_lbl].max()}
            xlim["min"], xlim["max"] = xlim["min"] - abs(0.1 * xlim["min"]), xlim["max"] + abs(0.1 * xlim["max"])
        if not ylim:
            ylim = {"min": _df[_predicted_lbl].min(), "max": _df[_predicted_lbl].max()}
            ylim["min"], ylim["max"] = ylim["min"] - abs(0.1 * ylim["min"]), ylim["max"] + abs(0.1 * ylim["max"])

        self.verbose("xlim={},ylim={}".format(xlim, ylim), func_name)
        ax = _df.plot(self.actual_lbl, self.predicted_lbl,
                      kind='scatter',
                      s=dot_size,
                      xlim=[xlim["min"], xlim["max"]],
                      ylim=[ylim["min"], ylim["max"]],
                      title="".format(title))

        ax.plot(np.linspace(xlim["min"], xlim["max"], 2),
                np.linspace(ylim["min"], ylim["max"], 2),
                linewidth=3, color='g')
        return ax

    def plot_confusion_matrix(self,
                              confusion_matrix,
                              classes_lst,
                              normalize=True,
                              title='Confusion Matrix',
                              number_formating="{:0.2f}"
                              ):
        """
        This function prints and plots a confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        func_name = self.plot_confusion_matrix.__name__

        _title = self._set_title(title)
        size = len(classes_lst) * 2.5
        plt.gcf().set_size_inches(cm2inch(size, size))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=self.cmap)
        plt.title(_title)
        plt.colorbar()
        tick_marks = np.arange(len(classes_lst))
        plt.xticks(tick_marks, classes_lst, rotation=45)
        plt.yticks(tick_marks, classes_lst)
        _confusion_matrix = confusion_matrix
        self.verbose("classes_lst={}".format(classes_lst), func_name)
        self.verbose("normalize={},title={} ".format(normalize, _title), func_name)
        if normalize:
            _confusion_matrix = _confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix:\n{}".format(_confusion_matrix))

        thresh = _confusion_matrix.max() / 2.0

        # def of internal unction : format_confusion_matrix_cell
        def format_confusion_matrix_cell(val, is_normlize):
            if is_normlize:
                txt = str(number_formating + "%").format(100.0 * round(val, 2))
            else:
                txt = str(number_formating).format(val)
            # print("{}-->{}".format(val, txt))
            return txt

        for i, j in itertools.product(range(_confusion_matrix.shape[0]), range(_confusion_matrix.shape[1])):
            txt = format_confusion_matrix_cell(val=_confusion_matrix[i, j], is_normlize=normalize)

            plt.text(j, i, txt,
                     horizontalalignment="center",
                     color="white" if _confusion_matrix[i, j] > thresh else "black", )

        plt.tight_layout()
        plt.ylabel("True label:{}".format(self.actual_lbl))
        plt.xlabel('Predicted label:{}'.format(self.predicted_lbl))

    def validation_curve(self, train_scores_df, valid_score_df):
        """
        source: http://scikit-learn.org/stable/modules/learning_curve.html
        Every estimator has its advantages and drawbacks.
        Its generalization error can be decomposed in terms of bias, variance and noise.
        The bias of an estimator is its average error for different training sets.
        The variance of an estimator indicates how sensitive it is to varying training sets.
        Noise is a property of the data.
        Bias and variance are inherent properties of estimators and we usually have to
        select learning algorithms and hyperparameters so that both bias and variance are
        as low as possible (see Bias-variance dilemma).
        Another way to reduce the variance of a model is to use more training data.
        However, you should only collect more training data if the true function is too
        complex to be approximated by an estimator with a lower variance.
        :param train_scores_df:
        :param valid_score_df:
        :return:
        """
        pass

    def show_values(self, pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857
        By HYRY
        '''

        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    def heatmap(self, auc, title="heatmap", xlabel="x", ylabel="y", xticklabels=[], yticklabels=[], figure_width=40,
                figure_height=20,
                correct_orientation=False, cmap='RdBu'):
        """

        Inspired by:
        - https://stackoverflow.com/a/16124677/395857
        - https://stackoverflow.com/a/25074150/395857
        :param auc:
        :param title:
        :param xlabel:
        :param ylabel:
        :param xticklabels:
        :param yticklabels:
        :param figure_width:
        :param figure_height:
        :param correct_orientation:
        :param cmap:
        :return:
        """

        func_name = self.plot_classification_report.__name__
        # Plot it out
        fig, ax = plt.subplots()
        c = ax.pcolor(auc, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(auc.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(auc.shape[1]) + 0.5, minor=False)

        # set tick labels
        # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, auc.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On, t.tick2On = False, False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On, t.tick2On = False, False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        self.show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            # resize
        fig = plt.gcf()
        # fig.set_size_inches(cm2inch(40, 20))
        # fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(cm2inch(figure_width, figure_height))
        return ax

    def plot_classification_report(self, classification_report, title='Classification report ', cmap='RdYlGn'):
        '''
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        '''
        func_name = self.plot_classification_report.__name__
        lines = classification_report.split('\n')

        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            print(v)
            plotMat.append(v)

        self.verbose('plotMat: {0}'.format(plotMat), func_name)
        self.verbose('support: {0}'.format(support), func_name)

        xlabel, ylabel = 'Metrics', 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = False
        ax = self.heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width,
                          figure_height,
                          correct_orientation, cmap=cmap)
        return ax
