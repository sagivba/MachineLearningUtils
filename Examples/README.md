# Usage Examples
## Datasets:
- [Iris dataset]
  * [scikit-learn site](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
  * [wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- Boston housing dataset
  * [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)

## creating iris-colored_acatter_matrix
You can create colored_acatter_matrix very easily ( 3 lines of code):
```python
    plotter=DataPlots(df=iris_df, ggplot=True)
    fig=plotter.colored_scatter_matrix(df=iris_df,colored_column_name="Target")
    fig.savefig("iris-colored_acatter_matrix.png")
```
[show me the complete code](iris-colored_acatter_matrix.py)
![iris-colored_acatter_matrix](iris-colored_acatter_matrix.png)

## simple example of iris dataset classification python
[show me the code](simple_usage-ModelUtils.py)
- load iris dataset
- create a DecisionTreeClassifier
- create a ModelUtils
```python
        prd_lbl, actl_lbl = "PrdictedIrisClass", "IrisClass"
        mu = ModelUtils(
                df            = iris_df,
                model           = tree_clf,
                predicted_lbl = "PrdictedIrisClass",
                actual_lbl    = "IrisClass"
                )
```
- split and train the model
```python
        mu.split_and_train()
```
- test the model
```python
        results_df = mu.test_model()
```
- evaluate results using plot_confusion_matrix
```python
        evp = EvaluationPlots(df=results_df, actual_lbl=mu.actual_lbl, predicted_lbl=mu.predicted_lbl)
        evp.plot_confusion_matrix(confusion_matrix=mu.confusion_matrix(), classes_lst=mu.model.classes_)
        plt.savefig("confusion_matrix.png", bbox_inches='tight')
```
    resulting this plot:
![evaluate results using plot_confusion_matrix](confusion_matrix.png)

- evaluate results using plot_confusion_matrix
```python
    cr = mu.classification_report(y_pred=results_df[prd_lbl], y_true=results_df[actl_lbl])
    evp.plot_classification_report(cr)
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
```
![evaluate results using plot_classification_report](classification_report.png)