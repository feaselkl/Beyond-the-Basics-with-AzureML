import argparse
import pandas as pd
import os
import json
from pathlib import Path
import mlflow
import azureml
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model

print("Register model...")
mlflow.start_run()

parser=argparse.ArgumentParser("register")
parser.add_argument("--model", type=str, help="Path to trained model")
parser.add_argument("--test_report", type=str, help="Path of model's test report")

args=parser.parse_args()

lines=[
    f"Model path: {args.model}",
    f"Test report path: {args.test_report}",
]

for line in lines:
    print(line)

run=Run.get_context()
ws=run.experiment.workspace

print('Loading models...')
model=mlflow.sklearn.load_model(Path(args.model))

print('Loading results...')
fname='results.json'
with open(Path(args.test_report) / fname, 'r') as fp:
    results=json.load(fp)
    
print('Saving models locally...')
root_model_path='trained_models'
os.makedirs(root_model_path, exist_ok=True)
mlflow.sklearn.save_model(model, root_model_path)

print("Registering the models...")
registered_model_name="ChicagoParkingTicketsCodeFirst"
model_description="Chicago Parking Tickets Boosted Tree Predictor - Code First Model"

registered_model=Model.register(model_path=root_model_path, 
                                  model_name=registered_model_name, 
                                  tags=results, 
                                  description=model_description, 
                                  workspace=ws)

mlflow.end_run()
print("Done!")
