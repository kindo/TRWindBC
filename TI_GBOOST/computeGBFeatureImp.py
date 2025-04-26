'''
Module Description:
------
This module is used to select the find the best parameter for the Gboost model and it also performs permutation features importance to identify the most important static features 

Return: 
	GBSelectFeatures.json
	GBParams.json
	GBFeatureImportance.json
	OBS_ScalingFactor.csv
Author: 
------
Freddy Houndekindo (freddy.houndekindo@inrs.ca)

'''


import pandas as pd
import numpy as np
import json

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm.contrib.concurrent import process_map


import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import StratifiedKFold

FEATURES = [	'GFSG1',
				'SG1-elev_percentile', 'SG1-dev_from_mean_elev', 'SG1-aspect',
				'SG1-slope', 'SG1-ruggedness_index',
				'SG1-standard_deviation_of_slope', 'SG1-total_curvature',
				'SG1-tangential_curvature', 'GFSG2', 'SG2-elev_percentile',
				'SG2-dev_from_mean_elev', 'SG2-aspect', 'SG2-slope',
				'SG2-ruggedness_index', 'SG2-standard_deviation_of_slope',
				'SG2-total_curvature', 'SG2-tangential_curvature', 'GFSG10',
				'SG10-elev_percentile', 'SG10-dev_from_mean_elev', 'SG10-aspect',
				'SG10-slope', 'SG10-ruggedness_index',
				'SG10-standard_deviation_of_slope', 'SG10-total_curvature',
				'SG10-tangential_curvature', 'GFSG20', 'SG20-elev_percentile',
				'SG20-dev_from_mean_elev', 'SG20-aspect', 'SG20-slope',
				'SG20-ruggedness_index', 'SG20-standard_deviation_of_slope',
				'SG20-total_curvature', 'SG20-tangential_curvature', 'GFSG50',
				'SG50-elev_percentile', 'SG50-dev_from_mean_elev', 'SG50-aspect',
				'SG50-slope', 'SG50-ruggedness_index',
				'SG50-standard_deviation_of_slope', 'SG50-total_curvature',
				'SG50-tangential_curvature', 'GFSG100', 'SG100-elev_percentile',
				'SG100-dev_from_mean_elev', 'SG100-aspect', 'SG100-slope',
				'SG100-ruggedness_index', 'SG100-standard_deviation_of_slope',
				'SG100-total_curvature', 'SG100-tangential_curvature', 'SRL_100m',
				'SRL_500m', 'SRL_1000m', 'SRL_5.0km', 'SRL_10.0km', 'SRL_20.0km',
				'Dcoast', '10ws_P05', '10ws_P50', '10ws_P95', 
				]

StaticPath = r'../data/staticFeatures.csv'
trainPath = r'../data/TRAIN_TEST/train.csv'
OBSPath = r'../data/ECCC_PARQUET/'
SFPath = r'../data/OBS_ScalingFactor.csv'
YEARS = np.arange(2008, 2018)

def compute_scaling_factor():
	filters = [('Year', 'in', YEARS)]
	data = pd.read_parquet(OBSPath, filters=filters, columns=['ERA5_WS_MS', 'WS_MS'])
	data = data.groupby('Climate_ID').mean()
	data = data.assign(OBS_SF = lambda x: x['WS_MS'] / x['ERA5_WS_MS'])
	data.to_csv(SFPath)

def generate_options(n):

	lr = [0.01, 0.02, 0.05, 0.1, 0.2]
	subsample = [0.5, 0.6, 0.7, 0.8, 0.9]
	max_depth = [2, 3, 4, 5, 6, 8]
	max_features = [0.1, 0.2, 0.3, 0.4, 0.5]
	n_estimators = [50, 100, 200, 400, 800, 1000]
	n_iter_no_change = [10, 20, 50, 100]
	min_samples_split = [0.001, 0.01, 0.02, 0.05, 0.1]


	paramOpts = {}

	for op in range(n):
		param = dict(loss='squared_error', 
					learning_rate=float(np.random.choice(lr)), 
					subsample=float(np.random.choice(subsample)), 
					max_depth=int(np.random.choice(max_depth)),
					max_features = float(np.random.choice(max_features)),
					n_estimators = int(np.random.choice(n_estimators)),
					n_iter_no_change = int(np.random.choice(n_iter_no_change)),
					min_samples_split = float(np.random.choice(min_samples_split)),
					criterion = 'friedman_mse',
					validation_fraction = 0.3,
					)
		paramOpts[f'opt_{op}'] = param

	with open(r'../../data/GBParams.json', 'w') as f:
		json.dump(paramOpts, f, indent=4)

def train_cv(data, features, target, params, nsplit=7, random_state=42, feature_importance=False):	
	
	kfold = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=random_state)

	pred_test = []
	stations_pred = []
	featureImportanceScore = []
	for k, (trainix, testix) in enumerate(kfold.split(data, data['Group'])):
		train = data.iloc[trainix]
		test = data.iloc[testix]

		x_test = test[features]
		y_test = test[[target]].values.flatten()
		
		x_train = train[features]
		y_train = train[[target]].values.flatten()

		model = GradientBoostingRegressor(**params)
		model.fit(x_train, y_train)

		pred_test += [np.clip(model.predict(x_test), 0, 1e5)]
		stations_pred += [test.index]

		if feature_importance:
			Feature_importance = permutation_importance(model, x_test, y_test, n_repeats=20,  scoring='r2')
			featureImportanceScore += [pd.DataFrame(Feature_importance.importances_mean, index=features, columns=[f'fold_{k}']) ]

	pred_test_df = pd.DataFrame((np.concatenate(pred_test)), index=np.concatenate(stations_pred), columns=['pred'])

	if feature_importance:
		return pred_test_df, featureImportanceScore
	
	return pred_test_df

def eval(op):

	""""
		Search for the best model hyperparameters using Random Search
	
	"""
	
	train = pd.read_csv(trainPath, index_col='Climate_ID')
	Static_data = pd.read_csv(StaticPath, index_col='Climate_ID')
	SF = pd.read_csv(SFPath, index_col='Climate_ID').loc[lambda x: x.index.isin(train.index)]

	data = SF.merge(Static_data[FEATURES], left_index=True, right_index=True)
	data = data.merge(train, left_index=True, right_index=True)

	with open(r'../../data/GBParams.json', 'r') as f:
		paramOpts = json.load(f)

	params = paramOpts[f'opt_{op}']
	
	target = 'OBS_SF'
	pred, FeaturesImportanceScore = train_cv(	data=data, 
												features=FEATURES, 
												target=target, 
												params=params, 
												nsplit=6, 
												random_state=op, 
												feature_importance=True, 
												)

	FeaturesImportanceScore = (pd.concat(FeaturesImportanceScore, axis=1)
											.median(axis=1)
											.sort_values(ascending=False)
								)

	obs = data.loc[pred.index, target]

	if (~np.isfinite(pred)).sum().values <= 0:
		eval_res = {'MAE' : mean_absolute_error(obs, pred), 'R2' : r2_score(obs, pred), 'MSE' : mean_squared_error(obs, pred)}
	else:
		print('Error:', op)
		eval_res = {'MAE' : np.nan, 'R2' : np.nan, 'MSE' : np.nan}
	return  {
				f'opt_{op}': 
							{	'Eval':eval_res, 
								'FeatureImportance':FeaturesImportanceScore.to_dict()
							}
			}

def main():

	compute_scaling_factor()
	generate_options(1000)
	ops = np.arange(1000)

	results = process_map(eval, ops, max_workers=10, chunksize=7)

	results = {k:v for d in results for k,v in d.items()}	
	
	with open(r'../data/GBFeatureImportance.json', 'w') as f:
		json.dump(results, f, indent=4)

	with open(r'../data/GBParams.json', 'r') as f:
		ParamsOpts = json.load(f)

	with open(r'../data/GBFeatureImportance.json', 'r') as f:
		PFIres = json.load(f)

		#	Select a bunch of good models. A threshold of R-squared = 0.36 was used
		Performance = (pd.DataFrame(
									{k: v['Eval'] for k, v in PFIres.items()}
									)
									.T.loc[lambda x: x.R2 > 0.36].sort_values('R2', ascending=False)
									)
		FeatureImportance = (pd.DataFrame(
											{k: v['FeatureImportance'] for k, v in PFIres.items() if k in Performance.index}
											).median(axis=1).sort_values(ascending=True)
											)

		#	Select the k best features
		selectedFeatures = FeatureImportance.sort_values(ascending=False).iloc[:11].index.to_list()
		opt_best = Performance.index[0]
		bestParams = ParamsOpts[opt_best]

	with open(r'../data/GBSelectFeatures.json', 'w') as f:
		json.dump({'selectedFeatures':selectedFeatures, 
			 		'bestParams':bestParams}, f, indent=4)


if __name__ == '__main__':
	main()