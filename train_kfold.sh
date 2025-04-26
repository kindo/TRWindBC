#   This script can be use to train and evaluate the models on the training set using cross-validation

# python ./TI_GBOOST/train_predict.py \
#         --mode 'kfold' \
#         --dataSRC './data/' \
#         --pred_folder './Predictions/TI_GBOOST' \

for k in {0..5}; 
    do
        python ./TR_model/train_predict.py \
            --model 'LSTM' \
            --data_path "./data/H5dataset/kfold/fold=$k" \
            --n_workers 1 \
            --in_memory 1 \
            --n_epochs 20 \
            --ckpt_folder './checkpoints/LSTM' \
            --pred_folder './Predictions/LSTM';
    done

for k in {0..5}; 
    do
        python ./TR_model/train_predict.py \
            --model 'Transformer' \
            --data_path "./data/H5dataset/kfold/fold=$k" \
            --n_workers 1 \
            --in_memory 1 \
            --n_epochs 20 \
            --ckpt_folder './checkpoints/Transformer' \
            --pred_folder './Predictions/Transformer';
    done

