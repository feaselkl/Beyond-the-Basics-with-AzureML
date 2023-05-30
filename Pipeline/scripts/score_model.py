import os
import glob
import mlflow
import pandas as pd
from azureml.core import Run
from azureml.core.model import Model

def init():
    global model

    print('Loading models...')
    run = Run.get_context()
    ws = run.experiment.workspace
    model_name = "ChicagoParkingTicketsCodeFirst"
    model_path = Model.get_model_path(model_name=model_name, _workspace=ws)
    # Load the model, it's input types and output names
    model = mlflow.sklearn.load_model(model_path)

def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")

    data = pd.concat(
        map(
            lambda fp: pd.read_csv(fp), mini_batch
        )
    )

    pred = model.predict(data)
    pred = pd.DataFrame(pred, columns=['PaymentIsOutstanding'])
    return data.assign(PaymentIsOutstanding=pred['PaymentIsOutstanding'])
