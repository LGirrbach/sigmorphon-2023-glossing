import os
import torch
import logging
import pandas as pd

from data import GlossingDataset
from pytorch_lightning import Trainer
from ctc_model import CTCGlossingModel
from containers import Hyperparameters
from morpheme_model import MorphemeGlossingModel
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


language_code_mapping = {
    "Arapaho": "arp",
    "Gitksan": "git",
    "Lezgi": "lez",
    "Natugu": "ntu",
    "Nyangbo": "nyb",
    "Tsez": "ddo",
    "Uspanteko": "usp",
}

model_type_mapping = {"ctc": CTCGlossingModel, "morph": MorphemeGlossingModel}


def _make_experiment_name(
    language: str,
    track: int,
    model_type: str,
    hyperparameters: Hyperparameters,
    trial: int,
):
    experiment_name = language_code_mapping[language]
    experiment_name = experiment_name + "-" + f"track{track}"
    experiment_name = experiment_name + "-" + f"model={model_type}"
    experiment_name = experiment_name + "-" + f"trial={trial}"

    hyperparameter_str = "-".join(
        [f"{param}={value}" for param, value in hyperparameters._asdict().items()]
    )
    experiment_name = experiment_name + "-" + hyperparameter_str
    return experiment_name


def _check_arguments(
    language: str,
    track: int,
    model_type: str,
    data_path: str,
    hyperparameters: Hyperparameters,
):
    assert isinstance(language, str) and language in language_code_mapping
    assert isinstance(track, int) and track in [1, 2]
    assert isinstance(model_type, str) and model_type in model_type_mapping
    assert os.path.isdir(data_path) and os.path.exists(data_path)

    assert (
        isinstance(hyperparameters.batch_size, int) and hyperparameters.batch_size >= 1
    )
    assert (
        isinstance(hyperparameters.num_layers, int) and hyperparameters.num_layers >= 1
    )
    assert (
        isinstance(hyperparameters.hidden_size, int)
        and hyperparameters.hidden_size >= 1
    )
    assert (
        isinstance(hyperparameters.dropout, float)
        and 0.0 <= hyperparameters.dropout <= 1.0
    )


def _make_train_path(language: str, track: int, data_path: str) -> str:
    language_code = language_code_mapping[language]
    return os.path.join(
        data_path, f"{language}/{language_code}-train-track{track}-uncovered"
    )


def _make_dev_path_uncovered(language: str, track: int, data_path: str) -> str:
    language_code = language_code_mapping[language]
    return os.path.join(
        data_path, f"{language}/{language_code}-dev-track{track}-uncovered"
    )


def _make_dev_path_covered(language: str, track: int, data_path: str) -> str:
    language_code = language_code_mapping[language]
    return os.path.join(
        data_path, f"{language}/{language_code}-dev-track{track}-covered"
    )


def _make_test_path(language: str, track: int, data_path: str) -> str:
    language_code = language_code_mapping[language]
    return os.path.join(
        data_path, f"{language}/{language_code}-test-track{track}-covered"
    )


def _make_dataset(
    language: str, track: int, data_path: str, batch_size: int
) -> GlossingDataset:
    train_file = _make_train_path(language, track, data_path)
    validation_file = _make_dev_path_uncovered(language, track, data_path)
    test_file = _make_test_path(language, track, data_path)

    dm = GlossingDataset(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        batch_size=batch_size,
    )

    return dm


def _make_model(
    model_type: str,
    dataset: GlossingDataset,
    track: int,
    hyperparameters: Hyperparameters,
):
    if model_type == "ctc":
        return CTCGlossingModel(
            source_alphabet_size=dataset.source_alphabet_size,
            target_alphabet_size=dataset.target_alphabet_size,
            hidden_size=hyperparameters.hidden_size,
            num_layers=hyperparameters.num_layers,
            dropout=hyperparameters.dropout,
        )
    elif model_type == "morph":
        learn_segmentation = track == 1
        classify_num_morphemes = track == 1
        return MorphemeGlossingModel(
            source_alphabet_size=dataset.source_alphabet_size,
            target_alphabet_size=dataset.target_alphabet_size,
            hidden_size=hyperparameters.hidden_size,
            num_layers=hyperparameters.num_layers,
            dropout=hyperparameters.dropout,
            learn_segmentation=learn_segmentation,
            classify_num_morphemes=classify_num_morphemes,
        )
    else:
        raise ValueError(f"Unknown Model Type: {model_type}")


def experiment(
    base_path: str,
    language: str,
    track: int,
    model_type: str,
    hyperparameters: Hyperparameters,
    data_path: str = "./data",
    verbose: bool = False,
    trial: int = 0,
):
    # Global Settings
    torch.set_float32_matmul_precision("medium")
    logging.disable(logging.WARNING)

    # Check Arguments
    _check_arguments(language, track, model_type, data_path, hyperparameters)

    # Make Experiment Name and Base Path
    experiment_name = _make_experiment_name(
        language, track, model_type, hyperparameters, trial
    )
    base_path = os.path.join(base_path, experiment_name)

    if os.path.exists(base_path):
        raise FileExistsError(f"Model Path {base_path} exists.")
    else:
        os.makedirs(base_path, exist_ok=True)

    # Prepare Data
    dm = _make_dataset(language, track, data_path, hyperparameters.batch_size)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Define Logger and Callbacks
    logger = pl_loggers.CSVLogger(
        save_dir=os.path.join(base_path, "logs"), name=experiment_name
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_path, "saved_models"),
        filename=experiment_name + "-{val_accuracy}",
        monitor="val_accuracy",
        save_last=True,
        save_top_k=1,
        mode="max",
        verbose=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", patience=3, mode="max", verbose=False
    )

    # Define Model
    model = _make_model(model_type, dm, track, hyperparameters)

    # Train Model
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        max_epochs=100,
        enable_progress_bar=verbose,
        log_every_n_steps=1,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_model_summary=verbose,
        callbacks=[early_stopping_callback, checkpoint_callback],
        min_epochs=1,
    )
    trainer.fit(model, dm)

    logs = pd.read_csv(
        os.path.join(base_path, "logs", experiment_name, "version_0", "metrics.csv")
    )
    best_val_accuracy = logs["val_accuracy"].max()
    return best_val_accuracy


if __name__ == "__main__":
    hparams = Hyperparameters(
        batch_size=2, num_layers=1, hidden_size=512, dropout=0.1, scheduler_gamma=1.0
    )
    res = experiment(
        base_path="./results",
        language="Natugu",
        track=1,
        model_type="morph",
        hyperparameters=hparams,
        verbose=True,
    )
    print(res)
