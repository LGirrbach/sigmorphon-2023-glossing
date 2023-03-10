import sys
import optuna
import logging

from experiment import experiment
from containers import Hyperparameters


batch_size_boundaries = {
    "Arapaho": [16, 128],
    "Gitksan": [2, 16],
    "Lezgi": [2, 64],
    "Nyangbo": [2, 64],
    "Tsez": [2, 64],
    "Uspanteko": [16, 128]
}


def hyperparameter_tuning(base_path: str, data_path: str, language: str, track: int, model_type: str,
                          num_trials: int = 1):
    batch_size_lower_bound = batch_size_boundaries[language][0]
    batch_size_upper_bound = batch_size_boundaries[language][1]

    # Define Optuna Logger
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Define Objective
    def objective(trial: optuna.Trial):
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        hidden_size = trial.suggest_int("hidden_size", 64, 512)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_size = trial.suggest_int(
            "batch_size", batch_size_lower_bound, batch_size_upper_bound
        )

        hyperparameters = Hyperparameters(
            batch_size=batch_size, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout
        )
        return experiment(
            base_path=base_path, data_path=data_path, language=language, track=track, model_type=model_type,
            hyperparameters=hyperparameters
        )

    # Setup Optuna
    study_name = f"glossing_tuning={language}-track{track}-{model_type}"
    storage_name = f"sqlite:///tuning/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="maximize"
    )
    study.optimize(objective, n_trials=num_trials)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(f"./tuning/{study_name}.csv")
