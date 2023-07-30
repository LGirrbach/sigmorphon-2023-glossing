import os
import torch
import shutil
import argparse
import pandas as pd

from data import GlossingDataset
from pytorch_lightning import Trainer
from ctc_model import CTCGlossingModel
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


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Glossing Model")
    parser.add_argument("--model", type=str, default="morph", choices=["ctc", "morph"])
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=list(language_code_mapping.keys()),
    )
    parser.add_argument(
        "--track",
        type=int,
        required=True,
        choices=[1, 2],
        help="Shared Task track. Can be 1 (closed) or 2 (open).",
    )
    parser.add_argument(
        "--layers", type=int, default=1, help="Num. layers of BiLSTM encoder."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability."
    )
    parser.add_argument(
        "--hidden", type=int, default=512, help="Hidden size of BiLSTM encoder."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Learning rate decay of exponential lr scheduler.",
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Max. num. epochs (early stopping always enabled).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Set torch matmul precision
    torch.set_float32_matmul_precision("high")

    # Parse arguments
    args = make_argument_parser()

    # Make experiment name
    name = f"glossing_{args.model}_{args.language}_{args.track}"

    # Make experiment directory
    shutil.rmtree("./results", ignore_errors=True)
    base_path = os.path.join("./results/", name)
    os.makedirs(base_path, exist_ok=True)

    # Make logger and callbacks
    logger = pl_loggers.CSVLogger(save_dir=os.path.join(base_path, "logs"), name=name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_path, "saved_models"),
        filename=name + "-{val_accuracy}",
        monitor="val_accuracy",
        save_last=True,
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", patience=3, mode="max", verbose=True
    )

    # Load data
    language = args.language
    track = args.track
    language_code = language_code_mapping[language]

    train_file = f"./data/{language}/{language_code}-train-track{track}-uncovered"
    validation_file = f"./data/{language}/{language_code}-dev-track{track}-uncovered"
    test_file = f"./data/{language}/{language_code}-dev-track{track}-covered"

    dm = GlossingDataset(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        batch_size=args.batch,
    )

    dm.prepare_data()
    dm.setup(stage="fit")

    # Make model and trainer
    if args.model == "ctc":
        model = CTCGlossingModel(
            source_alphabet_size=dm.source_alphabet_size,
            target_alphabet_size=dm.target_alphabet_size,
            num_layers=args.layers,
            dropout=args.dropout,
            hidden_size=args.hidden,
            scheduler_gamma=args.gamma,
        )
    elif args.model == "morph":
        model = MorphemeGlossingModel(
            source_alphabet_size=dm.source_alphabet_size,
            target_alphabet_size=dm.target_alphabet_size,
            num_layers=args.layers,
            dropout=args.dropout,
            hidden_size=args.hidden,
            scheduler_gamma=args.gamma,
            learn_segmentation=(track == 1),
            classify_num_morphemes=(track == 1),
        )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        log_every_n_steps=10,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[early_stopping_callback, checkpoint_callback],
        min_epochs=1,
    )

    # Train model
    trainer.fit(model, dm)
    # Load best model
    model.load_from_checkpoint(
        checkpoint_path=os.path.join(base_path, "saved_models", "last.ckpt")
    )

    # Load logs
    logs = pd.read_csv(
        os.path.join(base_path, "logs", name, "version_0", "metrics.csv")
    )
    best_val_accuracy = logs["val_accuracy"].max()
    print(f"Best validation accuracy: {100 * best_val_accuracy:.2f}")

    # Get Predictions
    dm.setup(stage="test")
    predictions = trainer.predict(model=model, dataloaders=dm.test_dataloader())

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
            dm.target_tokenizer.lookup_tokens(word_predictions)
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
                    "".join(dm.source_tokenizer.lookup_tokens(morpheme_indices))
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

    predictions_iterator = iter(decoded_predictions)

    # Write predictions
    with open(
        f"{args.language.lower()}_{args.model}_track{track}.prediction", "w"
    ) as pf:
        with open(test_file) as tf:
            for line in tf:
                if not line.startswith("\\g"):
                    pf.write(line)
                else:
                    prediction, segmentation = next(predictions_iterator)
                    if segmentation is not None:
                        pf.write("\\m " + segmentation + "\n")
                    pf.write("\\g " + prediction + "\n")
