# Getting Beyond the Basics with Azure Machine Learning

This repository provides the supporting code for my presentation entitled [Beyond the Basics with Azure ML](https://www.catallaxyservices.com/presentations/beyond-the-basics-with-azureml/).

## Generating Data

You will need an Azure SQL Database in order to run some of these tests.  In the `data` folder, there is a dacpac file which contains all of the tables and data you will need for this.

## Running the Code

### Basic Notebook

Import the notebook in the `Notebook` folder into Azure Machine Learning.  You will need to create a compute instance to run this.

### ML Pipeline

In order to run the ML pipeline notebooks locally, you will need to have the following installed on your machine:

* Python (preferably the Anaconda distribution)
* The Azure CLI
* The Azure ML Azure CLI extension
* Pip packages:  `azureml-core`, `azureml-pipeline`
* Visual Studio Code
* The Azure ML Visual Studio Code extension

## MLOps

The MLOps examples come from the [MLOps with Azure ML](https://github.com/microsoft/MLOpsPython/) GitHub repo.
