import os
import logging
import json
import numpy
import pickle
import joblib
import pandas as pd
from azureml.core.model import Model
from sklearn.tree import DecisionTreeRegressor


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    model_path = Model.get_model_path(model_name='ExpenseReportsPipelineModel')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    try:
        logging.info("Request received")
        data = pd.DataFrame(json.loads(raw_data)['data'])
        result = model.predict(data[["ExpenseCategoryID", "ExpenseYear"]])
        logging.info("Request processed")
        return result.tolist()
    except Exception as e:
        result = str(e)
        logging.error(result)
        return result