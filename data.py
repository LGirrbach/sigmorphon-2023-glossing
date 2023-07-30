import torch

from typing import List
from typing import Tuple
from typing import Dict
from typing import Union
from typing import Optional
from itertools import chain
from containers import Batch
from functools import partial
from torchtext.vocab import Vocab
from utils import nlp_pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from containers import GlossingFileData
from pytorch_lightning import LightningDataModule
from torchtext.vocab import build_vocab_from_iterator

RawDatapoint = Dict[str, Optional[Union[List[str], List[List[str]]]]]


def _make_empty_datapoint() -> RawDatapoint:
    return {"source": None, "target": None, "morphemes": None}


def _check_datapoint(datapoint: RawDatapoint) -> bool:
    source, target, morphemes = (
        datapoint["source"],
        datapoint["target"],
        datapoint["morphemes"],
    )
    if source is None:
        raise ValueError("Found Datapoint without Source.")

    if len(source) != len(target):
        return False

    if morphemes is not None and len(source) != len(morphemes):
        return False

    if any(len(wls) == 0 for wls in target):
        return False

    if morphemes is not None and any(
        len(ms) != len(wls) for ms, wls in zip(morphemes, target)
    ):
        return False

    if any(len(wls) > len(word) for word, wls in zip(source, target)):
        return False

    return True


def _datapoint_is_empty(datapoint: RawDatapoint) -> bool:
    return all(value is None for value in datapoint.values())


def read_glossing_file(file) -> GlossingFileData:
    track = 1 if "track1" in file else 2
    covered = "covered" in file and "uncovered" not in file

    raw_datapoints = [_make_empty_datapoint()]

    with open(file) as glossing_file:
        for line in glossing_file:
            line = line.strip()

            # Start New Datapoint on Empty Line
            if not line:
                raw_datapoints.append(_make_empty_datapoint())

            # Handle Source Lines
            elif line.startswith("\\t"):
                tokens = line[3:].split(" ")
                assert raw_datapoints[-1]["source"] is None
                raw_datapoints[-1]["source"] = tokens

            # Handle Morpheme Lines
            elif line.startswith("\\m"):
                tokens = line[3:].split(" ")
                morphemes = [token.split("-") for token in tokens]
                assert raw_datapoints[-1]["morphemes"] is None
                raw_datapoints[-1]["morphemes"] = morphemes

                # Replace Source with Canonical Segmentation
                assert raw_datapoints[-1]["source"] is not None
                raw_datapoints[-1]["source"] = tokens

            # Handle Glossing (=Target) Lines
            elif line.startswith("\\g"):
                word_labels = line[3:].strip().split(" ")
                word_labels = [label.strip() for label in word_labels if label.strip()]
                if not word_labels:
                    labels = None
                else:
                    labels = [word_label.split("-") for word_label in word_labels]

                assert raw_datapoints[-1]["target"] is None
                raw_datapoints[-1]["target"] = labels

            else:
                continue

    # Remove Empty Datapoints
    raw_datapoints = [
        datapoint for datapoint in raw_datapoints if not _datapoint_is_empty(datapoint)
    ]

    # Remove Corrupted Datapoints
    if not covered:
        raw_datapoints = [
            datapoint for datapoint in raw_datapoints if _check_datapoint(datapoint)
        ]

    # Check File Constraints
    if track == 2:
        assert all(datapoint["morphemes"] is not None for datapoint in raw_datapoints)

    if covered:
        assert all(datapoint["target"] is None for datapoint in raw_datapoints)
    else:
        assert all(datapoint["target"] is not None for datapoint in raw_datapoints)

    # Unpack Datapoints
    sources = [datapoint["source"] for datapoint in raw_datapoints]

    if track == 2:
        morphemes = [datapoint["morphemes"] for datapoint in raw_datapoints]
    else:
        morphemes = [None for _ in raw_datapoints]

    if not covered:
        targets = [datapoint["target"] for datapoint in raw_datapoints]
    else:
        targets = [None for _ in raw_datapoints]

    # Return Data
    return GlossingFileData(sources=sources, targets=targets, morphemes=morphemes)


def _make_source_sentence(
    source: List[str], sos_token: str = "[SOS]", eos_token: str = "[EOS]"
) -> List[str]:
    return [sos_token] + list(" ".join(source)) + [eos_token]


def _make_word_extraction_index(
    sources: List[List[str]], maximum_sentence_length: int, start_offset: int = 1
):
    word_extraction_index = []
    word_lengths = []
    word_batch_mapping = []
    words = []

    for i, source in enumerate(sources):
        start_index = i * maximum_sentence_length + start_offset
        for word in source:
            stop_index = start_index + len(word)
            word_indices = torch.arange(start_index, stop_index, dtype=torch.long)
            word_extraction_index.append(word_indices)
            word_lengths.append(word_indices.shape[0])
            word_batch_mapping.append(i)
            words.append(word)
            start_index = stop_index + 1

    word_extraction_index = nlp_pad_sequence(word_extraction_index)
    word_lengths = torch.tensor(word_lengths).long()

    return word_extraction_index, word_lengths, word_batch_mapping


def indices_to_tensor(indices: List[List[int]]) -> torch.Tensor:
    return nlp_pad_sequence([torch.tensor(idx).long() for idx in indices])


def _batch_collate(
    batch,
    source_tokenizer: Vocab,
    target_tokenizer: Vocab,
    sos_token: str = "[SOS]",
    eos_token: str = "[EOS]",
):
    sources, targets, morphemes = zip(*batch)

    # Encode Source Sentences (character level)
    make_source_sentence = partial(
        _make_source_sentence, sos_token=sos_token, eos_token=eos_token
    )
    source_sentences = [make_source_sentence(source) for source in sources]
    source_sentences = [source_tokenizer(source) for source in source_sentences]

    source_sentence_tensors = indices_to_tensor(source_sentences)
    source_sentence_lengths = torch.tensor(
        [len(source) for source in source_sentences]
    ).long()

    # Make Word Extraction Index
    maximum_sentence_length = source_sentence_tensors.shape[1]
    (
        word_extraction_index,
        word_lengths,
        word_batch_mapping,
    ) = _make_word_extraction_index(
        sources=sources, maximum_sentence_length=maximum_sentence_length
    )

    # Make Morpheme Extraction Index (In Case of Track 2)
    if all(ms is not None for ms in morphemes):
        maximum_word_length = max(word_lengths.tolist())
        morphemes_flat = list(chain.from_iterable(morphemes))

        (
            morpheme_extraction_index,
            morpheme_lengths,
            morpheme_word_mapping,
        ) = _make_word_extraction_index(
            morphemes_flat, maximum_word_length, start_offset=0
        )
    else:
        morpheme_extraction_index = None
        morpheme_lengths = None
        morpheme_word_mapping = None

    # Make Word Targets
    if all(target is not None for target in targets):
        word_targets = list(chain.from_iterable(targets))
        word_targets = [target_tokenizer(target) for target in word_targets]
        word_target_tensors = indices_to_tensor(word_targets)
        word_target_lengths = torch.tensor(
            [len(target) for target in word_targets]
        ).long()

        # Make Morpheme Targets
        morpheme_targets = list(chain.from_iterable(word_targets))
        morpheme_targets = torch.tensor(morpheme_targets).long()

    else:
        word_target_tensors = None
        word_target_lengths = None
        morpheme_targets = None

    return Batch(
        sentences=source_sentence_tensors,
        sentence_lengths=source_sentence_lengths,
        word_lengths=word_lengths,
        word_extraction_index=word_extraction_index,
        word_batch_mapping=word_batch_mapping,
        word_targets=word_target_tensors,
        word_target_lengths=word_target_lengths,
        morpheme_extraction_index=morpheme_extraction_index,
        morpheme_lengths=morpheme_lengths,
        morpheme_word_mapping=morpheme_word_mapping,
        morpheme_targets=morpheme_targets,
    )


class SequencePairDataset(Dataset):
    def __init__(self, dataset: GlossingFileData):
        super().__init__()
        self.dataset = dataset
        self._length = len(self.dataset.sources)
        assert len(self.dataset.sources) == len(self.dataset.targets)
        assert len(self.dataset.sources) == len(self.dataset.morphemes)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        return (
            self.dataset.sources[idx],
            self.dataset.targets[idx],
            self.dataset.morphemes[idx],
        )


class GlossingDataset(LightningDataModule):
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

    def __init__(
        self,
        train_file: str,
        validation_file: str,
        test_file: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file

        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            train_data = read_glossing_file(self.train_file)
            validation_data = read_glossing_file(self.validation_file)

            self.train_data = SequencePairDataset(train_data)
            self.validation_data = SequencePairDataset(validation_data)

            self.source_alphabet = set()
            self.source_alphabet.add(" ")
            for source in train_data.sources:
                for word in source:
                    self.source_alphabet.update(set(word))
            self.source_alphabet = list(sorted(self.source_alphabet))

            self.target_alphabet = set()
            for target in train_data.targets:
                for word_labels in target:
                    self.target_alphabet.update(set(word_labels))
            self.target_alphabet = list(sorted(self.target_alphabet))

            self.source_alphabet_size = len(self.source_alphabet) + 4
            self.target_alphabet_size = len(self.target_alphabet) + 4

            self.source_tokenizer = build_vocab_from_iterator(
                [[symbol] for symbol in self.source_alphabet],
                specials=self.special_tokens,
            )
            self.target_tokenizer = build_vocab_from_iterator(
                [[symbol] for symbol in self.target_alphabet],
                specials=self.special_tokens,
            )
            self.source_tokenizer.set_default_index(1)
            self.target_tokenizer.set_default_index(1)

            self._batch_collate = partial(
                _batch_collate,
                source_tokenizer=self.source_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        if stage == "test" or stage is None:
            self.test_data = SequencePairDataset(read_glossing_file(self.test_file))

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._batch_collate,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._batch_collate,
            num_workers=6,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._batch_collate,
            num_workers=6,
        )
