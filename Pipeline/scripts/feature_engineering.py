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


parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data", type=str, help="Name of folder we will write results out to")

args=parser.parse_args()

print("Performing feature engineering...")

lines=[
    f"Input data path: {args.input_data}",
    f"Output data path: {args.output_data}",
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

# Feature engineering steps
# year of issuance
df['Issued_date']=pd.to_datetime(df['Issued_date'])
df['Issued_year']=df['Issued_date'].dt.year

# categorize time based on hour of the day
# NOTE: cuts are right-aligned, so I'm starting with -1 to get the 0-6 hour range
hour_bins=[-1, 6, 10, 16, 19, np.inf ]
hour_names=['Overnight', 'Morning', 'Midday', 'AfterWork', 'Evening']
df['Time_of_day']=pd.cut(df['Issued_date'].dt.hour, bins=hour_bins, labels=hour_names)

# license plate origin
conds=[
    df['License_Plate_State'].isin(['IL']),
    df['License_Plate_State'].isin(['ON', 'ZZ', 'NB', 'AB', 'QU', 'MX', 'BC', 'MB', 'PE', 'NS', 'PQ', 'NF'])
]
choices=['In-state', 'Out-of-country']
df['License_plate_origin']=np.select(conds, choices, default='Out-of-state')

# vehicle type
conds=[
    df['Plate_Type'] == 'PAS',
    df['Plate_Type'] == 'TRK',
    df['Plate_Type'] == 'TMP'
]
choices=['PAS', 'TRK', 'TMP']
df['Vehicle_type']=np.select(conds, choices, default='Other')

# Write the results out for the next step.
print("Writing results out...")
df.to_csv((Path(args.output_data) / "FeatureEngineering.csv"), index=False)

print("Done!")
