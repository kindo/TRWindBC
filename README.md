# LSTM and Transformer-based framework for bias correction of ERA5 hourly wind speeds

This repo is the official Pytorch implementation of the paper: "[LSTM and Transformer-based framework for bias correction of ERA5 hourly wind speeds](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5125439)"

## Get Started
2. Set up the Python environment using **uv**. To set up the environment, you can run the bash script `./setup_env.sh`.
1. Download the pre-processed data from [] and unzip the file, e.g., `unzip TRwind.zip -d ./data`
2. Train the model using the bash scripts:
    - Cross-validation: `bash ./train_kfold.sh`
    - To reproduce the training and test experiment in the paper: `bash ./train_test.sh`

3. Evaluate the models using the Notebook provided in the folder `./evaluate`
