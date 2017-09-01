# MachineLearningUtils
***
Set of useful tools for machine learning projects

## Teasing examples:
### Plots
#### UsefulPlots.DataPlots.colored_scatter_matrix:
[(show me the code...)](Examples/iris-colored_acatter_matrix.py)
![UsefulPlots.colored_scatter_matrix](Examples/iris-colored_acatter_matrix.png)
#### UsefulPlots.EvaluationPlots.confusion:
[(Show me the code...)](Examples/modelutil_simple_usage.py)
![UsefulPlots.colored_scatter_matrix](Examples/iris-colored_acatter_matrix.png)
***
## ModelUtils
utils for easier skitlearn classifier handling
### classes:
- [ModelUtils](#class-ModelUtils)

## UsefulPlots
My toolbox of useful plots classes.
### classes:
- [DataPlots](#class-DataPlots) - plots for data exploration
- [EvaluationPlots](#class-EvaluationPlots) - Plots that helps to evaluate models
- [VisPlotPlayGround](#class-EvaluationPlots)

## DatasetsTools
### classes:
- [DatasetsTools](#class-DatasetsTools)

***

## module ModelUtils
### class ModelUtils
utils for easier skitlearn classifier handling

## module ModelUtils
Useful method for classifications :
Utils for easier skitlearn classifier handling:
- split_and_train
- test_model
- but you might want to use:
    - split_data_to_train_test
    - train_model

## module UsefulPlots
### class DataPlots
Plots for data exploration:
- colored_scatter -  Plot scatter of x vs y with color of third element
- colored_scatter_matrix - A matrix of colored_scatter

### class EvaluationPlots
Plots which helps to evaluate models
- predicted_vs_actual - This method creates sctter plot of
                        predicted values vs the actual valus.
- plots a confusion matrix. Normalization can be applied

### class VisPlotPlayGround
playground for visualization (color map and more...)
- show_colormap - Show the gradiant of cmap
- grayify_cmap  - Return a grayscale version of the colormap

## module DatasetsTools
mainly for unittests and demos
### class DatasetsTools
easyier skitlearn dataset exploration mainly for unittests and demos
- data_as_df


