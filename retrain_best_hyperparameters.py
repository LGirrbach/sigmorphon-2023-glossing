import json
import argparse

from experiment import experiment
from containers import Hyperparameters

languages = ["Arapaho", "Gitksan", "Lezgi", "Natugu", "Nyangbo", "Tsez", "Uspanteko"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Retrain Models with Best Hyperparameters")
    parser.add_argument("--basepath", default="./retrain_results")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--language", choices=languages, type=str)
    parser.add_argument("--track", type=int, choices=[1, 2])
    parser.add_argument("--model", type=str, choices=["ctc", "morph"])
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    language = args.language
    track = f"track{args.track}"
    model = args.model

    with open("./best_hyperparameters.json") as hf:
        hyperparameters = json.load(hf)
        hyperparameters = hyperparameters[language][track][model]
        hyperparameters = Hyperparameters(
            batch_size=int(hyperparameters["batch_size"]),
            num_layers=int(hyperparameters["num_layers"]),
            hidden_size=int(hyperparameters["hidden_size"]),
            dropout=hyperparameters["dropout"],
            scheduler_gamma=hyperparameters["scheduler_gamma"],
        )

    experiment(
        base_path=args.basepath,
        language=language,
        track=args.track,
        model_type=model,
        hyperparameters=hyperparameters,
        data_path=args.datapath,
        verbose=args.verbose,
        trial=args.trial,
    )
