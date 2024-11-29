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

All experiments with Deep Learning models were conducted on remote servers using an Nvidia RTX A6000 GPU (48 GB). Most experiments for Informer and PatchTST (except the ablation study with patching) can be run on less powerful GPUs (e.g., with 4 GB of memory).
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

## Experiments
NOTE: In notebooks 3, 4 set this to your cuda device, e. g.: cuda_device = "0".

1. The non-preprocessed dataset "time_series_60min_singleindex.csv" is in the folder ./datasets. The data used for our research originates from the Open Power System Data platform. It consists of hourly measurements on load, wind, and solar power generation (in megawatts) ranging from 2015 to the end of September 2020. The data was collected and preprocessed by the authors from the ENTSO-E Transparency platform.

The pre-processed datasets are provided in the ./datasets folder. They are generated to "DE_data.csv", "ES_data.csv", "FR_data.csv", "GB_data.csv", "IT_data.csv". However, you can take a look into the data preparation steps and data characteristics at .ipynb files: 1a, 1b and 1c.

2. The notebook 2.Base_models.ipynb provides a code for persistence forecast and (seasonal) ARIMA. The latter is fitted in univariate settings. Load and solar power generation columns are fitted with seasonal ARIMA (seasonal parameter 24 hours), and wind power generation columns are fitted with ARIMA without seasonal parameter.

3. 3a.Informer_PatchTST.ipynb is a notebook with main experiments for Informer and supervised PatchTST. The latter is run with 3 input windows and the best result forms the final PatchTST performance. 
Notebook 3b.PatchTST_self_supervised.ipynb presents the code for self-supervised PatchTST. It is execured with two stages: pre-training - reconstructing patches; and fine-tuning to the target task.
3c.Experiments_with_PatchTST.ipynb notebook include the ablation study experiments. Namely, individual components of PatchTST
are omitted while the others remain enabled, testing them one by one. These components include RevIN, Channel-independence and Patching.
In addition, there is a trend decomposition experiment.

4. TimeLLM_multi.ipynb notebook has a code for TimeLLM with multiprocessing on 4 GPUs. 

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)


## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

```bash
├── 1a.Data_selection.ipynb                                # First look at datasets and first data preparation steps (e.g. missing values imputation)
├── 1b.Data_analysis.ipynb                                 # EDA
├── 1c.Data_preparation.ipynb                              # Creation of country-based datasets
├── 2.Base_models.ipynb                                    # Implementation of persistence forecast and (seasonal) ARIMA
├── 3a.Informer_PatchTST.ipynb                             # Notebook to run Informer and supervised PatchTST for 3 input windows
├── 3b.PatchTST_self_supervised.ipynb                      # Notebook to run self-supervised PatchTST (pre-train & finetune)
├── 3c.Experiments_with_PatchTST.ipynb                     # Notebook to run ablation study: exclude RevIN/Channel-independence/Patching. + Time series trend decomposition (as in DLinear)
├── 4.TimeLLM_multi.ipynb                                  # Notebook to run TimeLLM with multiprocessing
├── 5.Results_comparison.ipynb                             # Final tables, figures and calculations
├── Appendix_1a.Scaler_choice.ipynb                        # APPENDIX: Notebook for scaler choise (StandardScaler, MinMaxScaler) for Germany
├── Appendix_1b.Scaler_choice_IT.ipynb                     # APPENDIX: for Italy
├── Appendix_1c.Scaler_choice_Comparison.ipynb             # APPENDIX: Comparison of unscaled evaluation metrics from models trained with different scalers. Plots with true and predicted values with different scalers for Germany
├── Appendix_1d.Scaler_choice_Comparison_IT.ipynb          # APPENDIX: for Italy
├── Appendix_2.Seasonality_proof.ipynb                     # APPENDIX: Seasonality with MSTL found from 45° line
├── datasets                                               # Directory with all datasets
├── logs                                                   # Directory with training logs for all DL models
├── NOTICE.txt                                             # Links to original PatchTST and TimeLLM implementations
├── PatchTST-main                                          # PatchTST directory (modified for purposes of this master's thesis)
├── requirements.txt                                       # Required dependencies
├── results                                                # .csv files with aggregated results (evaluation metrics for different models)
├── Time-LLM                                               # TimeLLM directory (modified for purposes of this master's thesis)
└── utils                                                  # Functions for EDA, and other helper functions
```
