import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
from typing import List, Dict

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(data: pd.DataFrame) -> np.ndarray:
    """
    Read the deployed model and a test dataset, calculate predictions.

    Parameters:
        data (pd.DataFrame): The test dataset containing 'lastmonth_activity', 
                             'lastyear_activity', and 'number_of_employees' columns.

    Returns:
        np.ndarray: An array containing the model predictions.
    """

    # Read the deployed model
    model = pickle.load(
        open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb")
    )

    # Keep relevant columns
    data = data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    y_pred = model.predict(data)

    return y_pred  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary() -> List[Dict[str, float]]:
    """
    Calculate summary statistics for the numeric independent variables in the dataset.

    Returns:
        List[Dict[str, float]]: A list of dictionaries, where each dictionary 
        represents the summary statistics for a numeric column. Each dictionary 
        has keys 'column', 'mean', 'median', and 'std' corresponding to the 
        column name, mean value, median value, and standard deviation value, 
        respectively.
    """

    # calculate summary statistics here
    filepath = os.path.join(dataset_csv_path, "finaldata.csv")
    df = pd.read_csv(filepath)
    df.drop(columns=["exited"], inplace=True)
    df_numeric = df.select_dtypes("number")

    statistics = []
    for col in df_numeric.columns:
        mean_val = df_numeric[col].mean()
        median_val = df_numeric[col].median()
        stddev_val = df_numeric[col].std()

        column_stats = {
            "column": col,
            "mean": mean_val,
            "median": median_val,
            "std": stddev_val,
        }

        statistics.append(column_stats)

    return statistics


def perc_missing_values() -> List[float]:
    """
    Calculate the percentage of missing values for each column in the dataset.

    Returns:
        List[float]: A list of percentages, where each value represents the 
        percentage of missing values in the corresponding column.
    """
    filepath = os.path.join(dataset_csv_path, "finaldata.csv")
    df = pd.read_csv(filepath)
    missing = df.isna().sum(axis=0)
    missing = (df.isna().sum(axis=0) / len(df)) * 100

    return missing.tolist()


##################Functions to get timings
def execution_time() -> List[float]:
    """
    Calculate the execution time of the 'ingestion.py' and 'training.py' scripts.

    Returns:
        List[float]: A list containing two timing values in seconds. The first 
        value represents the execution time of 'ingestion.py', and the second 
        value represents the execution time of 'training.py'.
    """
    # calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system("python ingestion.py")
    ingestion_time = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    os.system("python training.py")
    training_time = timeit.default_timer() - start_time
    return [
        ingestion_time,
        training_time,
    ]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list() -> pd.DataFrame:
    """
    Get a DataFrame containing outdated packages and their latest versions.

    Returns:
        pd.DataFrame: A DataFrame containing information about the outdated 
        packages. It has three columns: 'module', 'current', and 'latest'. 
        'module' contains the package name, 'current' contains the current 
        version of the package installed in the environment, and 'latest' 
        contains the latest version available for the package.
    """

    # get a list of
    with open("requirements.txt") as f:
        modules_list = [line.strip() for line in f]

    current_packages = [module.split("==") for module in modules_list]

    # newest versions
    outdated_versions = (
        subprocess.check_output(["pip", "list", "--outdated"], text=True)
        .strip()
        .split("\n")[2:]
    )
    package_latest_dict = {}
    for line in outdated_versions:
        package, _, latest, _ = line.split(maxsplit=3)
        package_latest_dict[package.strip()] = latest.strip()

    # Convert the list current modules in df
    df = pd.DataFrame(current_packages, columns=["module", "current"])
    df["latest"] = df["module"].map(package_latest_dict)
    return df


if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    data = test_df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    model_predictions(data)
    dataframe_summary()
    perc_missing_values()
    execution_time()
    outdated_packages_list()
