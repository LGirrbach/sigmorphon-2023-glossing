# Tü-CL at SIGMORPHON 2023: Straight-Through Gradient Estimation for Hard Attention
## Introduction

This repository contains code accompanying our system description paper [Tü-CL at SIGMORPHON 2023: Straight-Through Gradient Estimation for Hard Attention](https://aclanthology.org/2023.sigmorphon-1.17/) for the [Interlinear Glossing shared task](https://aclanthology.org/2023.sigmorphon-1.20/).
Note, that the code for the inflection model can be found [in this repository](https://github.com/LGirrbach/sigmorphon-2023-inflection).

The implementation of the joint segmentation and glossing model is in [morpheme_model.py](morpheme_model.py).
The segmentation model is in [morpheme_segmenter.py](morpheme_segmenter.py).

## Setup
Create a virtual environment, e.g. by using [Anaconda](https://docs.conda.io/en/latest/miniconda.html):
```
conda create -n glossing python=3.9 pip
```
Activate the environment:
```
conda activate glossing
```
Then, install the dependencies in [requirements.txt](requirements.txt):
```
pip install -r requirements.txt
```

Finally, place the shared task data in the repository, i.e. there should be a folder called `data`, which can be obtained from [the shared task's main repository](https://github.com/sigmorphon/2023glossingST).

## Train a model
To train a single model and get predictions for the corresponding test set, run
```
python main.py --language LANGUAGE --model MODEL --track TRACK
```
Only languages in the shared task dataset are supported. `MODEL` can be `ctc` for the BiLSTM with CTC loss baseline, or `morph` for the joint segmentation and glossing model. `TRACK` can be `1` for the closed track and `2` for the open track. For further hyperparameters, run
```
python main.py --help
```

## Hyperparameter tuning
To obtain best hyperparameters, you can use the script [hyperparameter_tuning.py](hyperparameter_tuning.py):
```
python hyperparameter_tuning.py \
    --language LANGUAGE \
    --model MODEL \
    --track TRACK \
    --trials TRIALS
```
Trials specifies the number of evaluated hyperparameter combinations. We used 50 for obtaining the hyperparameters provided in the file [best_hyperparameters.json](best_hyperparameters.json), which is included in this repository.

To retrain all models with our best hyperparameters, run
```
python retrain_best_hyperparameters.py
```
And to obtain predictions for the test data from the trained models, run
```
python predict_from_model.py
```

## Citation
If you use this code, consider citing our paper:
```
@inproceedings{girrbach-2023-tu,
    title = {T{\"u}-{CL} at {SIGMORPHON} 2023: Straight-Through Gradient Estimation for Hard Attention},
    author = "Girrbach, Leander",
    booktitle = "Proceedings of the 20th SIGMORPHON workshop on Computational Research in Phonetics, Phonology, and Morphology",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.sigmorphon-1.17",
    pages = "151--165",
}
```


