import argparse
from pathlib import Path
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import sys
import timeit
import numpy as np
from sklearn.model_selection import train_test_split


parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data_train", type=str, help="Name of folder we will write training results out to")
parser.add_argument("--output_data_test", type=str, help="Name of folder we will write test results out to")

args=parser.parse_args()

print("Performing feature selection...")

lines=[
    f"Input data path: {args.input_data}",
    f"Output training data path: {args.output_data_train}",
    f"Output test data path: {args.output_data_test}",
]

for line in lines:
    print(line)

print(os.listdir(args.input_data))

file_list=[]
for filename in os.listdir(args.input_data):
    print("Reading file: %s ..." % filename)
    with open(os.path.join(args.input_data, filename), "r") as f:
        input_df=pd.read_csv((Path(args.input_data) / filename))
        file_list.append(input_df)

# Concatenate the list of Python DataFrames
df=pd.concat(file_list)

# We will have a relatively smaller test dataset size
# but this is still ~200k rows and we're going to use some
# of the data for hyperparameter sweeps
train_df, test_df=train_test_split(df, test_size=0.2, random_state=11084)

# Write the results out for the next step.
print("Writing results out...")
train_df.to_csv((Path(args.output_data_train) / "TrainData.csv"), index=False)
test_df.to_csv((Path(args.output_data_test) / "TestData.csv"), index=False)

print("Done!")
