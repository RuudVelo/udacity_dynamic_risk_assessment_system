# udacity_dynamic_risk_assessment_system
Project 4 of the Udacity Machine Learning DevOps course

# Install project requirements
Create a conda environment and install using the requirements.txt. Activate your environment and also install the requests library using 

```
conda install -c anaconda requests
```

# Running the project

### Practice mode
Initially the project starts with using practicedata. For this make changes in the config.json file. Set "input_folder_path" to "practicedata" and "output_model_path" to "practicemodels" and save those. 

Also change the name of "confusion_matrix2.png" to "confusion_matrix.png" in the reporting.py file. In addition change the name of 'apireturns2.txt' to 'apireturns.txt' in the apicalls.py file. 

Then run in this sequence in your terminal:
1) ```python ingestion.py```
2) ```python training.py```
3) ```python scoring.py```
4) ```python deployment.py```
5) ```python reporting.py```
5) ```python app.py```

After app.py open a new terminal, activate your environment and run:
6) ```python apicalls.py```

The final report of predictions, model f1 score, summary statistics, package dependencies, training and ingestion times along with missing values can be found under "practicemodels/apireturns.txt"

### Production mode
For this make changes in the config.json file. Set "input_folder_path" to "sourcedata" and "output_model_path" to "models" and save those. 

Also change the name of "confusion_matrix.png" to "confusion_matrix2.png" in the reporting.py file. In addition change the name of 'apireturns.txt' to 'apireturns2.txt' in the apicalls.py file. Then run:

1) ```python ingestion.py```
2) ```python training.py```
3) ```python scoring.py```
5) ```python reporting.py```
5) ```python app.py```

After app.py open a new terminal, activate your environment and run:
6) ```python apicalls.py```

The last script (6) will generate a file called 'apireturns2.txt' in the 'models' folder. This is output on the new datasets (dataset3, dataset4) and a new trained model.

You can also run the fullprocess.py file after running the above. The fullprocess.py has the functionality to go through the complete pipeline of checking whether there is new data, if yes to ingest it. Then to check if model drift occurs based on a new trained model and then to train a new model if the f1 score is lower than before. If yes it would deploy the new model to production and run the reporting and apicalls.py with all the diagnostics again. What you will observe is what is described under **Important note**

# Important note
In the current setup the fullprocess actually stops in production after ingesting new files. Reason is that when using the old model (trained on dataset1, dataset2) and scoring it on the new data (dataset3, dataset4) the F1 score is actually higher (0.69) than the old model using the old data (0.57). When training a new model with the new data (dataset3, dataset4) and actually scoring it on the new data (datset3, dataset4) the f1 score is also lower, namely 0.3333. Hence, also in this case we don't want to used this newly trained model on new data. Hence, no model drift is detected and no new model deployment is done. 