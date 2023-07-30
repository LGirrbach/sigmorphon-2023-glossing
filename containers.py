from collections import namedtuple

GlossingFileData = namedtuple(
    "GlossingFileData", field_names=["sources", "targets", "morphemes"]
)
Batch = namedtuple(
    "Batch",
    [
        "sentences",
        "sentence_lengths",
        "word_lengths",
        "word_extraction_index",
        "word_batch_mapping",
        "word_targets",
        "word_target_lengths",
        "morpheme_extraction_index",
        "morpheme_lengths",
        "morpheme_word_mapping",
        "morpheme_targets",
    ],
)
Hyperparameters = namedtuple(
    "Hyperparameters",
    field_names=[
        "batch_size",
        "num_layers",
        "hidden_size",
        "dropout",
        "scheduler_gamma",
    ],
)
