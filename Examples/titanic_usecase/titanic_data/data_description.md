[source\(https://www.kaggle.com/c/titanic)

#Overview

The data has been split into two groups:

- training set (train.csv)
- test set (test.csv)


The training set should be used to build your machine learning models. 
For the training set, we provide the outcome (also known as the “ground truth”) 
for each passenger. Your model will be based on “features” like passengers’ gender and class. 
You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. 
For the test set, we do not provide the ground truth for each passenger. 
It is your job to predict these outcomes. 
For each passenger in the test set, use the model you trained to predict whether 
or not they survived the sinking of the Titanic.

# Data Dictionary

Variable     | Definition                                  | Key
------------ | --------------------------------------------|------
Survived     |	Survival                                   | 0 = No, 1 = Yes
Pclass	     |  Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd
Sex          |  Sex                                        |
Age	         |  Age in years                               |
SibSp        |  # of siblings / spouses aboard the Titanic |
Parch        |  # of parents / children aboard the Titanic |
Ticket	     |  Ticket number	                           |
Fare         |  Passenger fare	                           |
Cabin        |  Cabin number                               |
Embarked     |  Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton

## Variable Notes

- Pclass: A proxy for socio-economic status (SES)
        * 1st = Upper
        * 2nd = Middle
        * 3rd = Lower

- Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

- Sibsp: The dataset defines family relations in this way...
        * Sibling = brother, sister, stepbrother, stepsister
        * Spouse = husband, wife (mistresses and fianc?s were ignored)

- Parch: The dataset defines family relations in this way...
        * Parent = mother, father
        * Child = daughter, son, stepdaughter, stepson
        * Some children travelled only with a nanny, therefore parch=0 for them.