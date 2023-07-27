import os
import pickle
import subprocess

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import pandas as pd
import numpy as np
from sklearn import metrics

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']

source_files= os.listdir(input_folder_path)

##################Check and read new data
#first, read ingestedfiles.txt
continue_process = False

with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
    ingested_files = [line.strip().split('/')[-1] for line in file]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
continue_process = any(item not in ingested_files for item in source_files)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here

# If continue_process is True, some items in source_files are missing in ingested_files
if not continue_process:
    print("All files in source_files exist in ingested_files.")
    exit(0)

# If continue_process is False, all items in source_files exist in ingested_files
else:
    print("Not all files in source_files exist in ingested_files. Ingesting new files")
    ingestion.merge_multiple_dataframe()


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
    latest_score = [line.split('=')[-1].strip() for line in file]

data_path = os.path.join(output_folder_path,'finaldata.csv')
df = pd.read_csv(data_path)
y_pred = diagnostics.model_predictions(df)
y = df['exited']
f1_score_new = metrics.f1_score(y, y_pred)

print("Old f1 score based on old model and old data:" , float(latest_score[0]))
print("New f1 score based on old model and new data:", f1_score_new)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
model_drift = False
model_drift = True if f1_score_new < float(latest_score[0]) else False

if not model_drift:
    print("No model drift. The process is stopped")
    exit(0)
else:
    print("Model drift found. Re-train the model")

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
subprocess.call(['python', 'deployment.py'])


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
subprocess.call(['python', 'reporting.py'])
subprocess.call(['python', 'apicalls.py'])