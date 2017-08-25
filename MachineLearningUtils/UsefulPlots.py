import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib  import cm


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


    @classmethod
    def _set_fig(cls, fig=None, figsize=None):
        _figsize = cls._set_figsize(figsize)
        if fig:
            _fig = _fig
        else:
            _fig = plt.figure(figsize=_figsize)
            return _fig

    @classmethod
    def _set_subplot_info(cls, subplot_info=None):
        if subplot_info:
            _subplot_info = subplot_info
        else:
            _subplot_info = (1, 1, 1)
        return _subplot_info

    @classmethod
    def _set_figsize(cls, figsize=None):
        if figsize:
            _figsize = figsize
        else:
            _figsize = (8, 8)
        return _figsize

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
                sorted_uniq = sorted(z.unique())
                map_dict = dict([(k, i) for i, k in enumerate(sorted_uniq)])
                _z_col = z.map(lambda x: map_dict[x])
            scaler = MinMaxScaler(copy=True, feature_range=(30, 120))
            _Z = _z_col.to_frame()
            scaler.fit(_Z)
            _Z = scaler.transform(_Z)
            return _Z[:, 0]

        ### --- end defualt_color_func --- ###

        if color_func:
            _color_func = color_func
        else:
            _color_func = defualt_color_func

        return _color_func


class DataPlots(_BasePlot):
    """
    here  are some very useful plot for data exploration:
    colored_scatter plot scatter of x vs y whth color of third element

    """


    def colored_scatter(self, x, y, z2color, s=50, fig=None, subplot_info=None, figsize=None, color_func=None):
        """
        plot scatter of x vs y whth color of third element
        :param x: pandas series object
        :param y: pandas series object
        :param z2color: pandas series object
        :param fig    : plt.figure
        :param subplot_info: parameter of the fig.add_subplot
        :param color_func: input pandas series object - return pandas series object  of colors
        :return:
        """
        _color_func=self._set_color_func(color_func)

        _subplot_info=self._set_subplot_info(subplot_info)
        _figsize=self._set_figsize(figsize=figsize)
        _fig = self._set_fig(fig,_figsize)
        ax = _fig.add_subplot(*_subplot_info)
        ax.set_title("{} vs. {}".format(x.name,y.name), fontsize=14)
        ax.set_xlabel(x.name,fontsize=12)
        ax.set_ylabel(y.name,fontsize=12)
        ax.grid(True,linestyle='-',color='0.75')
        # scatter with colormap mapping to z value
        colors=_color_func(z2color)
        # print( pd.Series(colors).unique())
        ax.scatter(x,y,s=s,c=colors, marker = 'o', cmap = self.cmap);

        return ax

    def plot_column(self,data_column,fig=None,figsize=None):
        plt.style.use('ggplot')
        plot_dict = {
            "object" : [{"kind": "bar"}],
            "bool"   : [{"kind": "bar"}],
            "float64": [{"kind": "box"}, {"kind": "line"}, {"kind": "hist"}],
            "int64"  : [{"kind": "box"}, {"kind": "hist"}],
            "float32": [{"kind": "box"}, {"kind": "line"}],
            "int32"  : [{"kind": "box"}, {"kind": "hist"}],
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

    def scatterplot_matrix(self,data, names, figsize=(8, 8),**kwargs):
        """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
        against other rows, resulting in a nrows by nrows grid of subplots with the
        diagonal subplots labeled with "names".  Additional keyword arguments are
        passed on to matplotlib's "plot" command. Returns the matplotlib figure
        object containg the subplot grid."""
        _ncols, _nrecords = data.shape
        _figsize=self._set_figsize(figsize)
        fig, axes = plt.subplots(nrows=_ncols, ncols=_ncols, figsize=_figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)


        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                axes[x, y].plot(data[x], data[y], **kwargs)

        # Label the diagonal subplots...
        for i, label in enumerate(names):
            axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center')

        # Turn on the proper x or y axes ticks.
        for i, j in zip(range(_ncols), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            axes[i, j].yaxis.set_visible(True)

        return fig

    def axes_matrix(self,axes_matrix,fig=None,figsize=(12, 12),**kwargs):
        _fig=self._set_fig(fig,figsize)
        _ncols, _nrows = axes_matrix.shape


class VisPlotPlayGround(_BasePlot):
    """
    playground for visualization (color map and more...
    """

    def show_colormap(self,cmap,subplot_info=None):
        im = np.outer(np.ones(10), np.arange(100))
        _fig, axes = plt.subplots(2, figsize=(6, 1.5),
                               subplot_kw=dict(xticks=[], yticks=[]))
        _fig.subplots_adjust(hspace=0.1)
        axes[0].set_title("{}".format(cmap), fontsize=14)
        axes[0].imshow(im, cmap=cmap)
        axes[1].imshow(im, cmap=self.grayify_cmap(cmap))

        return _fig,axes

    def grayify_cmap(self, cmap):
        """Return a grayscale version of the colormap"""
        cmap = cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))

        # convert RGBA to perceived greyscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]

        return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
