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
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes

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

parent_dir="./config"
# Perform data preparation
# NOTE:  I would actually combine the feature-* and split-data steps together
# but have left them separate to show an example of multi-step pipelines
# NOTE:  Older versions of the AML SDK required 'path' instead of 'source' for the load_component() calls below.
replace_missing_values=load_component(source=os.path.join(parent_dir, "feature-replace-missing-values.yml"))
feature_engineering=load_component(source=os.path.join(parent_dir, "feature-engineering.yml"))
feature_selection=load_component(source=os.path.join(parent_dir, "feature-selection.yml"))
split_data=load_component(source=os.path.join(parent_dir, "split-data.yml"))
train_model=load_component(source=os.path.join(parent_dir, "train-model.yml"))
register_model=load_component(source=os.path.join(parent_dir, "register-model.yml"))

@pipeline(name="training_pipeline", description="Build a training pipeline")
def build_pipeline(raw_data):
    step_replace_missing_values=replace_missing_values(input_data=raw_data)
    step_feature_engineering=feature_engineering(input_data=step_replace_missing_values.outputs.output_data)
    step_feature_selection=feature_selection(input_data=step_feature_engineering.outputs.output_data)
    step_split_data=split_data(input_data=step_feature_selection.outputs.output_data)

    train_model_data=train_model(train_data=step_split_data.outputs.output_data_train,
                                   test_data=step_split_data.outputs.output_data_test,
                                   max_leaf_nodes=128,
                                   min_samples_leaf=32,
                                   max_depth=12,
                                   learning_rate=0.1,
                                   n_estimators=100)
    register_model(model=train_model_data.outputs.model_output, test_report=train_model_data.outputs.test_report)
    return { "model": train_model_data.outputs.model_output,
             "report": train_model_data.outputs.test_report }

def prepare_pipeline_job(cluster_name):
    # must have a dataset already in place
    cpt_asset=ml_client.data.get(name="ChicagoParkingTicketsFolder", version="1")
    raw_data=Input(type='uri_folder', path=cpt_asset.path)
    pipeline_job=build_pipeline(raw_data)
    # set pipeline level compute
    pipeline_job.settings.default_compute=cluster_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=True
    pipeline_job.display_name="train_pipeline"
    return pipeline_job

prepped_job=prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job, experiment_name="Chicago Parking Tickets Code-First")

print("Now look in the Azure ML Jobs UI to see the status of the pipeline job.  This will be in the 'Chicago Parking Tickets Code-First' experiment.")
