import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import time
import json
import pickle
import sys
import numpy as np
import pandas as pd
import urllib
from math import sqrt

import sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.base import clone
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn

def main(args):
    print(os.listdir(args.train_data))

    train_file_list=[]
    for filename in os.listdir(args.train_data):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.train_data, filename), "r") as f:
            input_df=pd.read_csv((Path(args.train_data) / filename))
            train_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    train_df=pd.concat(train_file_list)

    print(os.listdir(args.test_data))

    test_file_list=[]
    for filename in os.listdir(args.test_data):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.test_data, filename), "r") as f:
            input_df=pd.read_csv((Path(args.test_data) / filename))
            test_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    test_df=pd.concat(test_file_list)
    
    X_train, y_train=process_data(train_df)
    X_test, y_test=process_data(test_df)

    # train model
    params={
        'max_leaf_nodes': args.max_leaf_nodes,
        'min_samples_leaf': args.min_samples_leaf,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'validation_fraction': 0.1,
        'random_state': 11
        }
    
    model, results=train_model(params, X_train, X_test, y_train, y_test)
    
    print('Saving model...')
    mlflow.sklearn.save_model(model, args.model_output)
    
    print('Saving evauation results...')
    with open(Path(args.test_report) / 'results.json', 'w') as fp:
        json.dump(results, fp)
    
def process_data(df):
    numerical=['Per_capita_income', 'Percent_unemployed', 'Percent_without_diploma',
                 'Percent_households_below_poverty', 'Ward', 'ZIP', 'Police_District',
                 'Unit_ID', 'Violation_ID', 'Issued_year']
    categorical=['Time_of_day', 'License_plate_origin', 'Vehicle_type', 'Community_Name',
                   'Sector', 'Side', 'Neighborhood']
    label_column="PaymentIsOutstanding"
    all_columns=numerical + categorical + [label_column]
    
    df=df[all_columns]
    
    df=df[~df[label_column].isnull()]
    print("Number of non-null label rows:", len(df))
        
    X=df.drop(label_column, axis=1)
    y=df[label_column]
    
    for col in categorical:
        X[col]=X[col].astype('category')
        
    for col in numerical:
        X[col]=X[col].astype('float64')

    # return split data
    return X, y


def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    column_transformer=make_column_transformer(
            (make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(sparse=False)
            ), make_column_selector(dtype_include='category')),
            (make_pipeline(
                SimpleImputer(strategy='median'),
                MinMaxScaler()
            ), make_column_selector(dtype_exclude=['category']))
        )

    clf=GradientBoostingClassifier(**params)
    model=make_pipeline(column_transformer, clf)
    model=model.fit(X_train, y_train)
    
    y_preds=model.predict(X_test)

    accuracy=accuracy_score(y_test, y_preds)
    f1=f1_score(y_test, y_preds)
    f1_micro=f1_score(y_test, y_preds, average='micro')
    f1_macro=f1_score(y_test, y_preds, average='macro')
    precision=precision_score(y_test, y_preds)
    recall=recall_score(y_test, y_preds)
    roc_auc=roc_auc_score(y_test, y_preds)
    
    results={}
    results["accuracy"]=accuracy
    results["f1"]=f1
    results["f1_micro"]=f1_micro
    results["f1_macro"]=f1_macro
    results["precision"]=precision
    results["recall"]=recall
    results["roc_auc"]=roc_auc
    
    print(results)
    
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("f1", float(f1))
    mlflow.log_metric("f1_micro", float(f1_micro))
    mlflow.log_metric("f1_macro", float(f1_macro))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("roc_auc", float(roc_auc))

    # return model
    return model, results

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path of prepped train data")
    parser.add_argument("--test_data", type=str, help="Path of prepped test data")
    parser.add_argument('--max_leaf_nodes', type=int)
    parser.add_argument('--min_samples_leaf', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_report", type=str, help="Path of test_report")

    args=parser.parse_args()
    return args

# run script
if __name__ == "__main__":
    mlflow.start_run()
    
    args=parse_args()
    main(args)
    
    mlflow.end_run()
    print('Done!')
