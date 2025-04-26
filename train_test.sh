#   This script can be use to train the models and make prediction using the test data

python ./TI_GBOOST/train_predict.py \
        --mode 'test' \
        --dataSRC './data/' \
        --pred_folder './Predictions/TI_GBOOST' \
        --nbagging 100

python ./TR_model/train_predict.py \
    --model 'LSTM' \
    --data_path './data/H5dataset/train_test' \
    --n_workers 4 \
    --in_memory 1 \
    --n_epochs 20 \
    --ckpt_folder './checkpoints/LSTM' \
    --pred_folder './Predictions/LSTM'

python ./TR_model/train_predict.py \
    --model 'Transformer' \
    --data_path './data/H5dataset/train_test' \
    --n_workers 4 \
    --in_memory 1 \
    --n_epochs 20 \
    --ckpt_folder './checkpoints/Transformer' \
    --pred_folder './Predictions/Transformer'

