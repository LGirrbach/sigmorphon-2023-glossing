import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from itertools import chain
from containers import Batch
from torch.optim import AdamW
from utils import make_mask_2d
from bilstm import BiLSTMEncoder
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR


class CTCGlossingModel(LightningModule):
    def __init__(
        self,
        source_alphabet_size: int,
        target_alphabet_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_size: int = 128,
        scheduler_gamma: float = 1.0,
    ):
        super().__init__()
        self.source_alphabet_size = source_alphabet_size
        self.target_alphabet_size = target_alphabet_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.scheduler_gamma = scheduler_gamma

        self.save_hyperparameters()

        self.embeddings = nn.Embedding(
            num_embeddings=self.source_alphabet_size,
            embedding_dim=self.embedding_size,
            padding_idx=0,
        )
        self.encoder = BiLSTMEncoder(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            projection_dim=self.hidden_size,
        )
        self.classifier = nn.Linear(self.hidden_size, self.target_alphabet_size)
        self.ctc = nn.CTCLoss()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), weight_decay=0.0, lr=0.001)
        scheduler = ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]

    def get_prediction_scores(
        self, sentences: Tensor, sentence_lengths: Tensor, word_extraction_index: Tensor
    ):
        # Encode Sentences
        char_embeddings = self.embeddings(sentences)
        char_encodings = self.encoder(char_embeddings, sentence_lengths)
        char_predictions = self.classifier(char_encodings)
        # char_predictions: shape [batch x sentence length x target alphabet size]
        char_predictions = char_predictions.reshape(-1, self.target_alphabet_size)

        # Make Word Extraction Index
        num_words, chars_per_word = word_extraction_index.shape
        word_extraction_index_flat = word_extraction_index.flatten()
        word_prediction_scores = torch.index_select(
            char_predictions, dim=0, index=word_extraction_index_flat
        )
        word_prediction_scores = word_prediction_scores.reshape(
            num_words, chars_per_word, self.target_alphabet_size
        )

        return word_prediction_scores

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        prediction_scores = self.get_prediction_scores(
            batch.sentences, batch.sentence_lengths.cpu(), batch.word_extraction_index
        )

        log_probs = torch.log_softmax(prediction_scores, dim=-1)
        log_probs = log_probs.transpose(0, 1)

        targets = batch.word_targets
        loss = self.ctc(
            log_probs, targets, batch.word_lengths, batch.word_target_lengths
        )
        if torch.isinf(loss):
            return torch.tensor(0.0, requires_grad=True)

        return loss

    def evaluation_step(self, batch: Batch):
        prediction_scores = self.get_prediction_scores(
            batch.sentences, batch.sentence_lengths.cpu(), batch.word_extraction_index
        )
        predicted_indices = torch.argmax(prediction_scores, dim=-1)
        prediction_mask = make_mask_2d(batch.word_lengths)
        predicted_indices = torch.masked_fill(
            predicted_indices, mask=prediction_mask, value=0
        ).long()

        predicted_indices = predicted_indices.cpu().tolist()
        predicted_indices = [
            [idx for idx in predictions if idx != 0]
            for predictions in predicted_indices
        ]

        targets = batch.word_targets.cpu().tolist()
        targets = [[idx for idx in target if idx != 0] for target in targets]

        correct = [
            prediction == target
            for prediction, target in zip(predicted_indices, targets)
        ]
        return correct

    def validation_step(self, batch: Batch, batch_idx: int):
        return self.evaluation_step(batch=batch)

    def validation_epoch_end(self, outputs) -> None:
        correct = list(chain.from_iterable(outputs))

        accuracy = np.mean(correct)
        self.log("val_accuracy", 100 * accuracy)

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        prediction_scores = self.get_prediction_scores(
            batch.sentences, batch.sentence_lengths.cpu(), batch.word_extraction_index
        )
        predicted_indices = torch.argmax(prediction_scores, dim=-1)
        prediction_mask = make_mask_2d(batch.word_lengths)
        predicted_indices = torch.masked_fill(
            predicted_indices, mask=prediction_mask, value=0
        ).long()

        predicted_indices = predicted_indices.cpu().tolist()
        predicted_word_labels = [
            [idx for idx in predictions if idx != 0]
            for predictions in predicted_indices
        ]

        predicted_sentence_labels = [[] for _ in range(batch.sentences.shape[0])]
        for word_labels, sentence_idx in zip(
            predicted_word_labels, batch.word_batch_mapping
        ):
            predicted_sentence_labels[sentence_idx].append(word_labels)

        return predicted_sentence_labels, None
