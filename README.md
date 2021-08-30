# DeepExplanation: Model Explanation for Deep Learning of Shout Crisis Text Conversations

## Requirements

* lime 0.2.0.1
* matplotlib 3.4.2
* numpy 1.20.3
* scikit-learn 0.24.2
* seaborn 0.11.1
* pandas 1.3.1
* plotly 5.1.0
* python 3.9.5
* pytorch 1.9.0
* tqdm 4.49.0
* transformers 4.9.1

## Preprocessing
Data cleaning and preliminary processing can be found in `Preprocessing.ipynb`. Further preprocessing for each specific model is found with in the notebooks for training that respective model. A library to further facilitate data processing was developed by the former Masters student Daniel Cahn and can be found in `label_processor.py`. Other preliminary data exploration, including visualisations, can be found in `metadata.ipynb`.

## Models
Preprocessing data into the necessary format and training for each model are split into the following files: 

* `Basic_features_rf.ipynb`: basic features Random Forest model
* `TF-IDF.ipynb`: TF-IDF Random Forest model
* `bigbird_training.ipynb`: Masked Language Modelling (MLM) and single-label RoBERTa and Big Bird
* `multilabel_bigbird.ipynb`: multilabel RoBERTa and Big Bird

## LIME
Conversations for LIME explanations were selected in `lime_samples.ipynb`. Explanations were generated in `bigbird_lime_exps.ipynb`. `Explanations.ipynb` contains the aggregate explanations, while `interactive_LIME_exps.ipynb` visualises the explanations for individual conversations. Generated explanations for the individual conversations can be found in the folder `saved/bb_exps/*`. 
