# Analysis of modern AI models for time series forecasting in energy domain

**Type:** Master's Thesis 

**Author:** Valentyna Riabchuk

**1st Examiner:** Prof. Dr. Stefan Lessmann 

**2nd Examiner:** Dr. Alona Zharova 


![results](/Results_table.png)

## Table of Content

- [Abstract](#abstract)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Abstract

Long-sequence time series forecasting plays a crucial role in the energy sector. It provides future values for metrics that cannot be planned, relying on patterns from historical data. In our work, we apply the new PatchTST and TimeLLM models to three domains: load, solar power generation, and wind power generation, and compare them to the SOTA Informer model, (seasonal) ARIMA, and persistence forecasts across three forecasting horizons: 24, 96, and 168 hours, and five countries: Germany, the United Kingdom, Spain, France, and Italy. In addition, we perform ablation experiments on components of the PatchTST architecture. 
The results show that PatchTST outperforms all models in MAE across all countries and prediction lengths, but (S)ARIMA still outperforms other models in RMSE for 24-hour forecasts and for France and Italy in 96-hour forecasts. The ablation study revealed that all components of PatchTST contribute to its forecasting accuracy, with RevIN being the most important for data with significant scale shifts between train and test datasets. RevIN also helps mitigate daily, weekly, and seasonal variations during the training. Therefore, the model forecasts equally well for all days of the week and seasons. Channel independence allows the model to benefit from longer input windows and to learn local temporal patterns from individual features. Patching enhances prediction accuracy by extracting local information effectively and significantly speeds up the training process. TimeLLM with the GPT-2 backbone cannot surpass PatchTST. It is the second-best model because it incorporates advantageous components from PatchTST.

**Keywords**: Long-sequence time series forecasting, PatchTST, TimeLLM, Load, Solar power generation, Wind power generation, TSO

## Working with the repo

### Dependencies

The code was run on Python 3.11. The dependencies require Python >=3.8.

### Hardware requirements

All experiments with Deep Learning models were conducted on remote servers using an Nvidia RTX A6000 GPU (48 GB). Most experiments for Informer and PatchTST (excluding the patching ablation study) can be run on less powerful GPUs (e.g., with 4 GB of memory).
TimeLLM was executed using 4 Nvidia RTX A6000 GPUs in multiprocessing mode, with each GPU requiring 7 GB of memory. However, it can be run as a single process on one GPU. The line for this option is provided in the code.

### Setup

1. Clone this repository

2. Create an virtual environment and activate it
```
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

(Here is an example from SMART_HOME_N_ENERGY, [Appliance Level Load Prediction](https://github.com/Humboldt-WI/dissertations/tree/main/SMART_HOME_N_ENERGY/Appliance%20Level%20Load%20Prediction) dissertation)

```bash
├── 1a.Data_selection.ipynb                                # First look at datasets and first data prepparation steps (e.g. missing values)
├── 1b.Data_analysis.ipynb                                 # EDA
├── 1c.Data_preparation.ipynb                              # Creation of country-based datasets
├── 2.Base_models.ipynb                                    # Implementation of persistence forecast and (seasonal) ARIMA
├── 3a.Informer_PatchTST.ipynb                             # Basic e
├── 3b.PatchTST_self_supervised.ipynb
├── 3c.Experiments_with_PatchTST.ipynb
├── 4.TimeLLM_multi.ipynb
├── 5.Results_comparison.ipynb
├── Appendix_1a.Scaler_choice.ipynb
├── Appendix_1b.Scaler_choice_IT.ipynb
├── Appendix_1c.Scaler_choice_Comparison.ipynb
├── Appendix_1d.Scaler_choice_Comparison_IT.ipynb
├── Appendix_2.Seasonality_proof.ipynb
├── datasets                                               # Directory with all datasets
├── logs                                                   # Directory with training process for all models stored as log files
├── NOTICE.txt
├── PatchTST-main
├── requirements.txt
├── results
├── stored_elements
├── Time-LLM
└── utils              
```
