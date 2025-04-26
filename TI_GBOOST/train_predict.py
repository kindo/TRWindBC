'''
Module Description:
------
This module is used to train the time invariant gradient boosting model (TI-GBOOST)

Author: 
------
Freddy Houndekindo (freddy.houndekindo@inrs.ca)

'''

import os
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--mode', type=str, default='kfold', help='train modes: kfold or test')
parser.add_argument('--dataSRC', type=str, default='./data/')
parser.add_argument('--pred_folder', type=str, default='./pred', help='Folder to save the predictions')
parser.add_argument('--nbagging', type=int, default=100, help='Number of bagging ensemble')

args = parser.parse_args()

assert args.mode in ['kfold', 'test'], 'mode must be either kfold or test'
dataSRC = args.dataSRC
pred_folder = args.pred_folder
nbagging = args.nbagging

ParamPath = f'{dataSRC}GBParams.json'
FeaturesPath = f'{dataSRC}GBSelectFeatures.json'
traintestPath = f'{dataSRC}TRAIN_TEST'
StaticFeaturesPath = f'{dataSRC}staticFeatures.csv'
ScalingFactorPath = f'{dataSRC}OBS_ScalingFactor.csv'
kfoldPath = f'{dataSRC}kfolds.json'

def train_func(data, params, train, test, features, target):
    
    x_train = data.loc[train, features].values
    y_train = data.loc[train, [target]].values.flatten()

    x_test = data.loc[test, features].values

    model = GradientBoostingRegressor(**params)
    model.fit(x_train, y_train)

    y_pred = (pd.DataFrame({'predictions': model.predict(x_test), 
                                'Climate_ID':test})
                .set_index('Climate_ID')
                )
    
    return y_pred

def main():
    with open(FeaturesPath, 'r') as f:
        features_param = json.load(f)
        features = features_param['selectedFeatures']
        params = features_param['bestParams']

    data = pd.read_csv(StaticFeaturesPath, index_col='Climate_ID')
    ScalingFactor = pd.read_csv(ScalingFactorPath, index_col='Climate_ID')
    data = ScalingFactor.merge(data, left_index=True, right_index=True)

    target = 'OBS_SF'

    os.makedirs(pred_folder, exist_ok=True)

    if args.mode == 'kfold':

        with open(kfoldPath, 'r') as f:
            folds = json.load(f) 

        Ypred= []
        for k in range(len(folds.keys())):
            train = folds[f'fold_{k}']['train']
            test = folds[f'fold_{k}']['test']
            ypred = train_func(data, params, train, test, features, target)
            Ypred += [ypred]
        Ypred = pd.concat(Ypred, axis=0)

        Ypred.to_csv(os.path.join(pred_folder, 'kfold.csv'))
    
    if args.mode == 'test':
        train = pd.read_csv(os.path.join(traintestPath, 'train.csv'), index_col='Climate_ID').index.values
        test = pd.read_csv(os.path.join(traintestPath, 'test.csv'), index_col='Climate_ID').index.values

        Ypred = []
        for _ in range(nbagging):
            ypred = train_func(data, params, train, test, features, target)
            Ypred += [ypred]

        Ypred = pd.concat(Ypred, axis=1).mean(axis=1).rename('predictions')
        Ypred.to_csv(os.path.join(pred_folder, 'test.csv'))

if __name__ == '__main__':
    main()





    
    

