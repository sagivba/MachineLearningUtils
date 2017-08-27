import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

__version__ = '0.0.4'

class _BasePlot():
    """
    collecton of class methos and more for the plots class below
    """

    def __init__(self, df=None, ggplot=True, cmap=cm.OrRd):
        self.df = df
        self.ggplot = ggplot
        if ggplot:
            self.use_ggplt()
        self.cmap = cmap

    def use_ggplt(self):
        plt.style.use('ggplot')

    def _set_df(self, df=None):
        if df is None:
            _df = df
        else:
            _df = self.df
        return _df

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
            :return:
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

        return _ax

    def plot_column(self, data_column, fig=None, figsize=None):
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
                axes[i].set_title("{}-{}-{}".format(_data.name, _kind, 'count'))
                y_val.plot(kind='bar', title=_data.name + ' count', ax=axes[i])
            elif str(_data.dtype) not in ['object', 'bool']:
                _data.plot(kind=_kind, ax=axes[i])
            else:
                return None
        return _fig

    def colored_scatter_matrix(self, df, colored_column_name, figsize=(12, 12), **kwargs):
        """
        Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
        against other rows, resulting in a nrows by nrows grid of subplots with the
        diagonal subplots labeled with "names".  Additional keyword arguments are
        passed on to matplotlib's "plot" command. Returns the matplotlib figure
        object containg the subplot grid."""
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

        _figsize = self._set_figsize(figsize)
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
