import os
import torch
import shutil
import pandas as pd

from data import GlossingDataset
from pytorch_lightning import Trainer
from ctc_model import CTCGlossingModel
from morpheme_model import MorphemeGlossingModel
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    name = "glossing_test"
    shutil.rmtree("./results", ignore_errors=True)
    base_path = os.path.join("./results/", name)
    logger = pl_loggers.CSVLogger(save_dir=os.path.join(base_path, "logs"), name=name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(base_path, "saved_models"), filename=name + "-{val_accuracy}", monitor="val_accuracy",
        save_last=True, save_top_k=1, mode="max", verbose=True
    )
    early_stopping_callback = EarlyStopping(monitor="val_accuracy", patience=3, mode="max", verbose=True)

    language_code_mapping = {
        "Arapaho": "arp",
        "Gitksan": "git",
        "Lezgi": "lez",
        "Nyangbo": "nyb",
        "Tsez": "ddo",
        "Uspanteko": "usp"
    }

    language = "Lezgi"
    track = 1
    language_code = language_code_mapping[language]

    train_file = f"./data/{language}/{language_code}-train-track{track}-uncovered"
    validation_file = f"./data/{language}/{language_code}-dev-track{track}-uncovered"
    test_file = f"./data/{language}/{language_code}-dev-track{track}-covered"

    dm = GlossingDataset(
        train_file=train_file,
        validation_file=validation_file,
        test_file=test_file,
        batch_size=22
    )

    dm.prepare_data()
    dm.setup(stage="fit")

    model = MorphemeGlossingModel(
        source_alphabet_size=dm.source_alphabet_size, target_alphabet_size=dm.target_alphabet_size,
        num_layers=1, dropout=0.42, hidden_size=496, scheduler_gamma=0.98,
        learn_segmentation=(track == 1), classify_num_morphemes=(track == 1)
    )
    trainer = Trainer(
        accelerator="gpu", devices=1, gradient_clip_val=1.0, max_epochs=25, enable_progress_bar=True,
        log_every_n_steps=10, logger=logger, check_val_every_n_epoch=1,
        callbacks=[early_stopping_callback, checkpoint_callback], min_epochs=1
    )

    trainer.fit(model, dm)
    model.load_from_checkpoint(checkpoint_path=os.path.join(base_path, "saved_models", "last.ckpt"))

    logs = pd.read_csv(os.path.join(base_path, "logs", name, "version_0", "metrics.csv"))
    best_val_accuracy = logs["val_accuracy"].max()
    print(best_val_accuracy)

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
    for sentence_prediction, sentence_segmentation in zip(sentence_predictions, sentence_segmentations):
        decoded_sentence_prediction = [
            dm.target_tokenizer.lookup_tokens(word_predictions) for word_predictions in sentence_prediction
        ]
        decoded_sentence_prediction = ["-".join(word_predictions) for word_predictions in decoded_sentence_prediction]
        decoded_sentence_prediction = " ".join(decoded_sentence_prediction)

        if sentence_segmentation is not None:
            decoded_sentence_segmentation = [
                ["".join(dm.source_tokenizer.lookup_tokens(morpheme_indices)) for morpheme_indices in token_indices]
                for token_indices in sentence_segmentation
            ]
            decoded_sentence_segmentation = ["-".join(morphemes) for morphemes in decoded_sentence_segmentation]
            decoded_sentence_segmentation = " ".join(decoded_sentence_segmentation)
        else:
            decoded_sentence_segmentation = None

        decoded_predictions.append((decoded_sentence_prediction, decoded_sentence_segmentation))

    predictions_iterator = iter(decoded_predictions)

    with open(f"{language_code}_track{track}.prediction", "w") as pf:
        with open(test_file) as tf:
            for line in tf:
                if not line.startswith("\\g"):
                    pf.write(line)
                else:
                    prediction, segmentation = next(predictions_iterator)
                    if segmentation is not None:
                        pf.write("\\m " + segmentation + "\n")
                    pf.write("\\g " + prediction + "\n")
