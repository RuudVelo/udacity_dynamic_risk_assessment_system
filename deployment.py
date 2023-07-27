import os
import json
import shutil


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

output_folder_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])

####################function for deployment
def store_model_into_pickle() -> None:
    """
    Copy the latest pickle file, the latestscore.txt value, and the ingestedfiles.txt file
    into the deployment directory.

    Parameters:
        None

    Returns:
        None
    """
    shutil.copy(
        os.path.join(output_model_path, "trainedmodel.pkl"), prod_deployment_path
    )
    shutil.copy(
        os.path.join(output_model_path, "latestscore.txt"), prod_deployment_path
    )
    shutil.copy(
        os.path.join(output_folder_path, "ingestedfiles.txt"), prod_deployment_path
    )


if __name__ == "__main__":
    store_model_into_pickle()
