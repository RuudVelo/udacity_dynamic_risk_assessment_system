import pickle
import pandas as pd
import numpy as np
import json
import os
from typing import Any, List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from diagnostics import model_predictions


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])

# Function to generate confusion matrix plot
def plot_confusion_matrix(y_test: Any, y_pred: Any, model_path: str) -> None:
    """
    Generate a confusion matrix plot and save it to a file.

    Args:
        y_test (Any): True labels from the test dataset.
        y_pred (Any): Predicted labels obtained from the model.
        model_path (str): The path where the confusion matrix plot will be saved.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(model_path, "confusion_matrix2.png"), bbox_inches="tight")
    plt.close()


##############Function for reporting
def score_model():
    """
    Calculate a confusion matrix using the test data and the deployed model, and write
    the confusion matrix plot to the workspace.
    """
    filepath = os.path.join(test_data_path, "testdata.csv")
    df = pd.read_csv(filepath)
    X_test = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = df[["exited"]]

    y_pred = model_predictions(X_test)

    return plot_confusion_matrix(y_test, y_pred, model_path)


if __name__ == "__main__":
    score_model()
