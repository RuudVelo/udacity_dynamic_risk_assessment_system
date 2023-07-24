import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])

# Function to generate confusion matrix plot
def plot_confusion_matrix(y_test, y_pred, dataset_csv_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(dataset_csv_path, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    filepath = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(filepath)
    y_test = df[['exited']]

    y_pred = model_predictions()

    return plot_confusion_matrix(y_test, y_pred, dataset_csv_path)

if __name__ == '__main__':
    score_model()
