import os
import sys
import optuna
import logging
import argparse

from experiment import experiment
from containers import Hyperparameters


batch_size_boundaries = {
    "Arapaho": [16, 128],
    "Gitksan": [2, 16],
    "Lezgi": [2, 64],
    "Natugu": [2, 64],
    "Nyangbo": [2, 64],
    "Tsez": [2, 64],
    "Uspanteko": [16, 128],
}


def hyperparameter_tuning(
    base_path: str,
    data_path: str,
    language: str,
    track: int,
    model_type: str,
    num_trials: int = 1,
):
    batch_size_lower_bound = batch_size_boundaries[language][0]
    batch_size_upper_bound = batch_size_boundaries[language][1]

    # Define Optuna Logger
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Define Objective
    def objective(trial: optuna.Trial):
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        hidden_size = trial.suggest_int("hidden_size", 64, 512)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.9, 1.0)
        batch_size = trial.suggest_int(
            "batch_size", batch_size_lower_bound, batch_size_upper_bound
        )

        hyperparameters = Hyperparameters(
            batch_size=batch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            scheduler_gamma=scheduler_gamma,
        )
        return experiment(
            base_path=base_path,
            data_path=data_path,
            language=language,
            track=track,
            model_type=model_type,
            hyperparameters=hyperparameters,
        )

    # Setup Optuna
    os.makedirs("./tuning", exist_ok=True)
    study_name = f"glossing_tuning={language}-track{track}-{model_type}"
    # Skip if exists
    if os.path.exists(f"./tuning/{study_name}.csv"):
        return
    elif os.path.exists(f"./tuning/{study_name}.db"):
        os.remove(f"./tuning/{study_name}.db")

    storage_name = f"sqlite:///tuning/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="maximize"
    )
    study.optimize(objective, n_trials=num_trials)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(f"./tuning/{study_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Tuning")
    parser.add_argument("--basepath", default="./results")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument(
        "--language", choices=list(batch_size_boundaries.keys()), type=str
    )
    parser.add_argument("--track", type=int, choices=[1, 2])
    parser.add_argument("--model", type=str, choices=["ctc", "morph"])
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()

    hyperparameter_tuning(
        base_path=args.basepath,
        data_path=args.datapath,
        language=args.language,
        track=args.track,
        model_type=args.model,
        num_trials=args.trials,
    )
