from sklearn.model_selection import train_test_split
from azureml.core import Run

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import pandas as pd
from math import sqrt
import os

# dataset object from the run
run = Run.get_context()
dataset = run.input_datasets["prepared_expensereport_ds"]

# split dataset into train and test set
(train_dataset, test_dataset) = dataset.random_split(percentage=0.8, seed=2416)

# load datasets into Pandas dataframes
data_train = train_dataset.to_pandas_dataframe()
data_test = test_dataset.to_pandas_dataframe()

# train the decision tree regression model
reg = DecisionTreeRegressor()
reg.fit(data_train[["ExpenseCategoryID", "ExpenseYear"]], data_train[["Amount"]].values.ravel())

# test results
pred = pd.DataFrame({"AmountPrediction": reg.predict(data_test[["ExpenseCategoryID", "ExpenseYear"]]) })
# Concatenate testing data with predictions
testdf = pd.concat([data_test, pred], axis=1)


# generate metrics
# Calculate the overall root mean squared error
rmse = sqrt(mean_squared_error(testdf["Amount"], testdf["AmountPrediction"]))
run.log('RMSE', rmse)
# Calculate the per-employee-and-expense-category RMSE
employees = testdf.groupby(['EmployeeName', 'ExpenseCategory'])
for cat, grp in employees:
    empname, expcat = cat
    rmse = sqrt(mean_squared_error(grp["Amount"], grp["AmountPrediction"]))
    rescat = ('{}, {}, RMSE'.format(empname, expcat))
    run.log(rescat, rmse)

# save the model
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs("./outputs", exist_ok=True)
model_file_name = 'outputs/model.pkl'
joblib.dump(value = reg, filename = model_file_name)