import os

from tqdm import tqdm
from experiment import _make_dataset
from pytorch_lightning import Trainer
from ctc_model import CTCGlossingModel
from experiment import _make_test_path
from morpheme_model import MorphemeGlossingModel


language_code_mapping = {
    "Arapaho": "arp",
    "Gitksan": "git",
    "Lezgi": "lez",
    "Natugu": "ntu",
    "Nyangbo": "nyb",
    "Tsez": "ddo",
    "Uspanteko": "usp",
}

code_language_mapping = {
    code: language for language, code in language_code_mapping.items()
}


def decode_predictions(predictions, source_tokenizer, target_tokenizer):
    sentence_predictions = []
    sentence_segmentations = []
    for batch_prediction in predictions:
        batch_prediction, batch_segmentation = batch_prediction
        sentence_predictions.extend(batch_prediction)
        if batch_segmentation is not None:
            sentence_segmentations.extend(batch_segmentation)
        else:
            sentence_segmentations.extend([None for _ in batch_prediction])

    decoded_predictions = []
    for sentence_prediction, sentence_segmentation in zip(
        sentence_predictions, sentence_segmentations
    ):
        decoded_sentence_prediction = [
            target_tokenizer.lookup_tokens(word_predictions)
            for word_predictions in sentence_prediction
        ]
        decoded_sentence_prediction = [
            "-".join(word_predictions)
            for word_predictions in decoded_sentence_prediction
        ]
        decoded_sentence_prediction = " ".join(decoded_sentence_prediction)

        if sentence_segmentation is not None:
            decoded_sentence_segmentation = [
                [
                    "".join(source_tokenizer.lookup_tokens(morpheme_indices))
                    for morpheme_indices in token_indices
                ]
                for token_indices in sentence_segmentation
            ]
            decoded_sentence_segmentation = [
                "-".join(morphemes) for morphemes in decoded_sentence_segmentation
            ]
            decoded_sentence_segmentation = " ".join(decoded_sentence_segmentation)
        else:
            decoded_sentence_segmentation = None

        decoded_predictions.append(
            (decoded_sentence_prediction, decoded_sentence_segmentation)
        )

    return decoded_predictions


def write_predictions(
    predictions,
    language_code: str,
    track: int,
    model_type: str,
    trial: int = 1,
    base_path: str = "./predictions",
    data_path: str = "./data",
):
    os.makedirs(base_path, exist_ok=True)

    predictions_iterator = iter(predictions)
    prediction_file_name = os.path.join(
        "./predictions",
        f"{language_code}_track{track}_{model_type}_trial{trial}.prediction",
    )
    test_file_name = _make_test_path(
        language=code_language_mapping[language_code], track=track, data_path=data_path
    )

    with open(prediction_file_name, "w") as pf:
        with open(test_file_name) as tf:
            for line in tf:
                if not line.startswith("\\g"):
                    pf.write(line)
                else:
                    prediction, segmentation = next(predictions_iterator)
                    # if segmentation is not None:
                    #    pf.write("\\m " + segmentation + "\n")
                    pf.write("\\g " + prediction + "\n")


def get_predictions(
    path_to_model: str,
    language: str,
    track: int,
    model_type: str,
    data_path: str = "./data",
    verbose: bool = False,
):
    # Load Data
    dm = _make_dataset(language, track, data_path, 16)
    dm.prepare_data()
    dm.setup(stage="fit")
    dm.setup(stage="test")

    # Load Model
    if model_type == "ctc":
        model = CTCGlossingModel.load_from_checkpoint(checkpoint_path=path_to_model)
    elif model_type == "morph":
        model = MorphemeGlossingModel.load_from_checkpoint(
            checkpoint_path=path_to_model
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Create Trainer
    # Train Model
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=verbose,
        logger=False,
    )

    # Get predictions
    predictions = trainer.predict(model=model, dataloaders=dm.test_dataloader())
    predictions = decode_predictions(
        predictions, dm.source_tokenizer, dm.target_tokenizer
    )
    return predictions


def parse_model_name(model_name: str):
    entries = model_name.split("-")
    language_code = entries[0]
    track = int(entries[1][-1])
    model = entries[2].split("=")[1]
    trial = int(entries[3].split("=")[1])

    return language_code, track, model, trial


def get_predictions_from_retrained_models(base_path: str = "./retrain_results"):
    for retrained_model in os.listdir(base_path):
        language_code, track, model_type, trial = parse_model_name(retrained_model)

        path_to_saved_models = os.path.join(base_path, retrained_model, "saved_models")
        best_checkpoint = [
            checkpoint
            for checkpoint in os.listdir(path_to_saved_models)
            if checkpoint != "last.ckpt"
        ]
        best_checkpoint = best_checkpoint[0]
        path_to_best_checkpoint = os.path.join(path_to_saved_models, best_checkpoint)

        predictions = get_predictions(
            path_to_model=path_to_best_checkpoint,
            language=code_language_mapping[language_code],
            track=track,
            model_type=model_type,
        )

        write_predictions(
            predictions=predictions,
            language_code=language_code,
            track=track,
            model_type=model_type,
            trial=trial,
        )


if __name__ == "__main__":
    get_predictions_from_retrained_models()
