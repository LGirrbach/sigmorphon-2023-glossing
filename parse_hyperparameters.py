import os
import json
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


def remove_parameter_name_prefix(parameter_name: str) -> str:
    if not parameter_name.startswith("params_"):
        return parameter_name
    else:
        return parameter_name[len("params_") :]


def parse_tuning_results() -> None:
    result_path = "./tuning"
    best_hyperparameters = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    tuning_result_files = [
        file for file in os.listdir(result_path) if file.endswith(".csv")
    ]
    for tuning_result_file in tqdm(tuning_result_files):
        results = pd.read_csv(
            os.path.join(result_path, tuning_result_file), index_col=0
        )
        results = results.drop(["number", "state"], axis=1)
        results = results[results["value"] == results["value"].max()]
        results = results.reset_index().iloc[0].to_dict()
        results = {
            remove_parameter_name_prefix(param_name): value
            for param_name, value in results.items()
        }
        results = {
            key: value
            for key, value in results.items()
            if key not in ["index", "value"]
        }
        language, track, model = tuning_result_file[:-4].split("=")[1].split("-")
        best_hyperparameters[language][track][model] = results

    with open("best_hyperparameters.json", "w") as hsf:
        json.dump(best_hyperparameters, hsf)


if __name__ == "__main__":
    parse_tuning_results()
