# boston_housing_rm

The deployed web app is live at https://iris-streamlit-rm.herokuapp.com/

This web app predicts the median price/value of a house located in Boston area as a function of its input parameters : CRIM,ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT.

* **CRIM:** per capita crime rate by town.
* **ZN:** proportion of residential land zoned for lots over 25,000 sq.ft.
* **INDUS:** proportion of non-retail business acres per town.
* **CHAS:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
* **NOX:** nitric oxides concentration (parts per 10 million).
* **RM:** average number of rooms per dwelling.
* **AGE:** proportion of owner-occupied units built prior to 1940.
* **DIS:** weighted distances to five Boston employment centres.
* **RAD:** index of accessibility to radial highways.
* **TAX:** full-value property-tax rate per $10,000.
* **PTRATIO:** pupil-teacher ratio by town.
* **B:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
* **LSTAT:** lower status of the population.


The variable to predict is MEDV:

* **MEDV:** Median value of owner-occupied homes in $1000.

**Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/**
*This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The web app was built in Python using the following libraries:

* streamlit
* pandas
* scikit-learn
* shap
