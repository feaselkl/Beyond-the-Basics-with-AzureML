import sys
import os
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from random import randrange
import urllib
from urllib.parse import urlencode

import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Model,
    AmlCompute,
    Data,
    BatchRetrySettings,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction

from azure.ai.ml.sweep import (
    Choice,
    Uniform
)

# NOTE:  set your workspace name here!
workspace_name="CSAzureML"
# NOTE:  if you do not have a cpu-cluster already, we will create one
# Alternatively, change the name to a CPU-based compute cluster
cluster_name="cpu-cluster"

# NOTE:  for local runs, I'm using the Azure CLI credential
# For production runs as part of an MLOps configuration using
# Azure DevOps or GitHub Actions, I recommend using the DefaultAzureCredential
#ml_client=MLClient.from_config(DefaultAzureCredential())
ml_client=MLClient.from_config(AzureCliCredential())
ws=ml_client.workspaces.get(workspace_name)

# Make sure the compute cluster exists already
try:
    cpu_cluster=ml_client.compute.get(cluster_name)
    print(
        f"You already have a cluster named {cluster_name}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure Machine Learning compute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is available.\
    # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

    cpu_cluster=AmlCompute(
        name=cluster_name,
        # Azure Machine Learning Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster=ml_client.compute.begin_create_or_update(cpu_cluster)


# Ensure that there is an endpoint for batch scoring
endpoint_name="chicago-parking-tickets-batch"
try:
    endpoint=ml_client.batch_endpoints.get(endpoint_name)
    print(f"You already have an endpoint named {endpoint_name}, we'll reuse it as is.")
except Exception:
    print("Creating a new batch endpoint")
    endpoint=BatchEndpoint(name=endpoint_name, description="Batch scoring endpoint for Chicago Parking Ticket payment status")
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    endpoint=ml_client.batch_endpoints.get(endpoint_name)
    print(f"Endpoint name:  {endpoint.name}")

# Retrieve the parking tickets model
model=ml_client.models.get(name="ChicagoParkingTicketsCodeFirst", version="1")
print("Retrieved model.")

# Get the correct environment
deployment_name="cpt-batch-deployment"
try:
    deployment=ml_client.batch_deployments.get(name=deployment_name, endpoint_name=endpoint_name)
    print(f"You already have a deployment named {deployment_name}; we'll reuse it as is.")
except Exception:
    print("No deployment exists--creating a new deployment")
    environment=ml_client.environments.get(name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu", version="33")
    deployment=ModelBatchDeployment(
        name="cpt-batch-deployment",
        description="Batch scoring of Chicago Parking Ticket payment status",
        endpoint_name=endpoint.name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(code="scripts", scoring_script="score_model.py"),
        compute=cluster_name,
        settings=ModelBatchDeploymentSettings(
            instance_count=2,
            max_concurrency_per_instance=2,
            mini_batch_size=10,
            output_action=BatchDeploymentOutputAction.APPEND_ROW,
            output_file_name="predictions.csv",
            retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
            logging_level="info",
        )
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    print("Created deployment.")
    # Now make the deployment the default for our endpoint
    endpoint.defaults.deployment_name=deployment.name
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print("Made deployment the default for this endpoint.")

# Prepare the dataset
data_path="data"
dataset_name="ChicagoParkingTicketsUnlabeled"
try:
    chicago_dataset_unlabeled=ml_client.data.get(dataset_name, label="latest")
    print("Dataset already exists.")
except Exception:
    print("No dataset exists--creating a new dataset")
    chicago_dataset_unlabeled=Data(
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        description="An unlabeled dataset for Chicago parking ticket payment status",
        name=dataset_name
    )
    ml_client.data.create_or_update(chicago_dataset_unlabeled)
    chicago_dataset_unlabeled=ml_client.data.get(dataset_name, label="latest")
    print("Dataset now exists.")

# NOTE: If you are getting a "BY_POLICY" error, make sure that your account is an Owner
# on the Azure ML workspace.  You must *explicitly* grant rights, even if you are the 
# subscription owner!
# Create a job to score the data
job=ml_client.batch_endpoints.invoke(endpoint_name=endpoint.name, input=Input(type=AssetTypes.URI_FOLDER, path=chicago_dataset_unlabeled.path))
# Wait for the job to finish
ml_client.jobs.stream(job.name)
# Download the results of the job
ml_client.jobs.download(name=job.name, output_name='score', download_path='./')
