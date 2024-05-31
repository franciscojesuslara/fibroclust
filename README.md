Characterization-fibro-ml-models
====

Repository for reproducibility of results for the paper "Characterization of fibromyalgia using the analysis of the central and autonomic nervous systems with machine learning models", which is focused on finding clusters in patients with fibromyalgia through using nonlinear reduction methods and clustering approaches.

## Installation and setup

To download the source code, you can clone it from the Github repository.
```console
git clone git@github.com:franciscojesuslara/fibroclust.git
```

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt 
```

To replicate the results, download data and put them into folder **data** folder. Specifically: 

## Preprocessing and Feature Extraction

The steps followed to preprocess the data and to extract features from the database are detailed in `features_extraction.ipynb`. However, due to privacy concerns, the data used in this project cannot be shared to protect the privacy and confidentiality of the participants involved.


## To obtain different results of data-driven models

To perform consensus clustering:
```console
python src/detect_groups.py
```


