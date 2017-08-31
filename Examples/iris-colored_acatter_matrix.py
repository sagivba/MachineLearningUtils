from sklearn import datasets
from MachineLearningUtils.UsefulPlots import DataPlots
from MachineLearningUtils.DatasetTools import DatasetsTools



def main():
    # load iris data into DataFrame
    iris_dtst = DatasetsTools(datasets.load_iris)
    iris_df = iris_dtst.data_as_df()
    print ("columns: {}".format(list(iris_df)))
    print (iris_dtst.info)
    plotter=DataPlots(df=iris_df, ggplot=True)
    fig=plotter.colored_scatter_matrix(df=iris_df,colored_column_name="Target")
    fig.savefig("iris-colored_acatter_matrix.png")

    return

if __name__ == '__main__':
    main()