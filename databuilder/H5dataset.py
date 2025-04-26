'''
Module Description:
------
This module is used to create the HDF5 dataset ready for the model training and evaluation

Author: 
------
Freddy Houndekindo (freddy.houndekindo@inrs.ca)

'''


import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from dataloader import Dataloader
from sklearn.model_selection import StratifiedKFold, train_test_split
import json

DATE_ORIGINE = r'1900-01-01 00:00'

Paths = {	'eraPath': 		r'../data/ERA5_PARQUET',
			'StaticPath': 	r'../data/staticFeatures.csv',
			'obsPath': 		r'../data/ECCC_PARQUET',
			}


eraFeatures = ['10u', '10v', '10ws', 'blh', 'sp', '2t'] 

staticFeatures = [	"10ws_P05",
					"10ws_P50",
					"10ws_P95",
					"GFSG1",
					"SG1-standard_deviation_of_slope",
					"SG10-dev_from_mean_elev",
					"SG10-tangential_curvature",
					"SG100-standard_deviation_of_slope",
					"SG2-standard_deviation_of_slope",
					"SRL_1000m",
					"SRL_500m"]

CONFIG = {	
			'obs_col':'WS_MS',
			'era_col':'ERA5_WS_MS',
			'Paths': Paths,
			'eraFeatures':eraFeatures,
			'staticFeatures':staticFeatures,
			'plen':24*5,
			'flen': 24,
			}

TRAIN = pd.read_csv(r'../data/TRAIN_TEST/train.csv')
TEST = pd.read_csv(r'../data/TRAIN_TEST/test.csv')

def build_kfold(n_splits):
	
	skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=45)

	folds = list(skfold.split(TRAIN, TRAIN['Group']))
	Climate_ID = TRAIN['Climate_ID'].values
	folds_dict = {f'fold_{k}': 
							{'train':Climate_ID[folds[k][0]].tolist(), 
							 'test':Climate_ID[folds[k][1]].tolist()
							 } 
					for k in range(len(folds))
					}
	
	with open(r'../data/kfolds.json', 'w') as f:
		json.dump(folds_dict, f, indent=4)

def get_bsize(config, Climate_ID, years):
	cfg = config | {	'years': years,
						'Climate_ID':Climate_ID, 
					}
	
	Loader = Dataloader(cfg)
	B = Loader.batch.shape[0]
	return B

def create_hdf5(B, path):

	with h5py.File(path, 'w') as f:
		#   ERA 
		C = len(CONFIG['eraFeatures'])
		T_era = CONFIG['plen'] + CONFIG['flen']
		_ = f.create_dataset('DynamicFeatures', (B, T_era, C), chunks=(100, T_era, C),  dtype=np.float32, compression="gzip", compression_opts=9)

		#   Topo
		C = len(CONFIG['staticFeatures'])
		_ = f.create_dataset('StaticFeatures', (B, C), chunks= (100, C), dtype=np.float32, compression="gzip", compression_opts=9)

		#   dates
		_ = f.create_dataset('DateFeatures', (B, T_era, 3), chunks= (100, T_era, 3), dtype=np.uint32, compression="gzip", compression_opts=9)

		#   Target
		_ = f.create_dataset('target', (B, CONFIG['flen']), chunks=(100, CONFIG['flen']), dtype=np.float32, compression="gzip", compression_opts=9)

		#   Meta
		_ = f.create_dataset('Dates', (B, ), dtype=np.uint32, compression="gzip", compression_opts=9)
		_ = f.create_dataset('Climate_ID', (B, ), dtype=np.dtype('S7'), compression="gzip", compression_opts=9)

	return 

def _build_dataset(year, config, Cids, hdf5Path, last_sample_size=0):

	cfg = config | {     	
							'years':year,
							'Climate_ID': Cids
							}


	
	loader = Dataloader(cfg)
	loader.build_dates()
	B = loader.batch.shape[0]

	with h5py.File(hdf5Path, 'r+') as f:

	##############################################################################
		#print('working static features')

		StaticFeatures = loader.load_static()
		ds = f['StaticFeatures']

		ds[last_sample_size  : last_sample_size + B, ...] = StaticFeatures
	   
		del StaticFeatures

	##############################################################################
		#print('working on dates')

		dates = loader.load_dates()
		ds = f['DateFeatures']   
		ds[last_sample_size :last_sample_size + B, ...] = dates

		del dates

	##############################################################################
		#print('Working on target')
		
		target = loader.load_target()
		ds = f['target']
		ds[last_sample_size : last_sample_size + B, ...] = target
		
		del target
	##############################################################################
		#print('Working on ERA5')

		era = loader.load_era()
		ds = f['DynamicFeatures']
		ds[last_sample_size : last_sample_size + B, ...] = era

		del era
		
	###############################################################################
		#print('saving climate_IDs')
		Climate_ID = loader.batch.index.get_level_values('Climate_ID').values.reshape(-1,)
		ds = f['Climate_ID']
		ds[last_sample_size : last_sample_size + B, ...] = Climate_ID.astype('S7')


		ds = f['Dates']
		dates = loader.batch['Date00h00'].values
		dates = (pd.to_datetime(dates) - pd.Timestamp(DATE_ORIGINE)).astype('timedelta64[h]').astype(int).values.reshape(-1, )
		ds[last_sample_size : last_sample_size + B, ...] = dates

	###############################################################################
	return last_sample_size + B

def buildH5Dataset(H5PATH, train_val, test, TRAIN_YEARS, TEST_YEARS, fold=None):
	
	os.makedirs(H5PATH, exist_ok=True)

	train, val = train_test_split(	train_val, 
										stratify=TRAIN.set_index('Climate_ID').loc[train_val, 'Group'].values, 
										test_size=0.3,
										random_state=45,
										)
	config = CONFIG | {'train_Climate_ID' : train}

	
	B_train = get_bsize(config, train, TRAIN_YEARS)
	B_val = get_bsize(config, val, TRAIN_YEARS)
	B_test = get_bsize(config, test, TEST_YEARS)

	print(	f'Train size={B_train}',
	   		f'Test size={B_test}',
			f'val size={B_val}'
		)
	
	if fold is not None:
		h5path = f'{H5PATH}fold={fold}/'
	else:
		h5path = H5PATH
	os.makedirs(h5path, exist_ok=True)

	trainPath = f'{h5path}train.h5'
	testPath = f'{h5path}test.h5'
	valPath = f'{h5path}val.h5'

	create_hdf5(B_train, trainPath)
	create_hdf5(B_test, testPath)
	create_hdf5(B_val, valPath)

	last_sample_size = 0
	for year in tqdm(TRAIN_YEARS, desc='train', leave=False):
		last_sample_size = _build_dataset([year], config, train, trainPath, last_sample_size)

	last_sample_size = 0
	for year in tqdm(TEST_YEARS, desc='test', leave=False):
		last_sample_size = _build_dataset([year], config, test, testPath, last_sample_size)

	last_sample_size = 0
	for year in tqdm(TRAIN_YEARS, desc='val', leave=False):
		last_sample_size = _build_dataset([year], config, val, valPath, last_sample_size)

	#	Save the configuration informations
	config = config | {'Train_years': TRAIN_YEARS} | {'Test_years': TEST_YEARS} \
					| {'val_Climate_ID' : val} | {'test_Climate_ID' : test}

	with open(f'{h5path}config.json', 'w') as f:
		json.dump(config, f, indent=4)

def main():
	nsplit = 6

	#build_kfold(nsplit)

	H5PATH = r'../data/H5dataset/train_test/'
	train_val = TRAIN['Climate_ID'].values.tolist()
	test = TEST['Climate_ID'].values.tolist()

	TRAIN_YEARS = np.arange(2008, 2018).tolist()
	TEST_YEARS = np.arange(2018, 2024).tolist()

	buildH5Dataset(H5PATH, train_val, test, TRAIN_YEARS, TEST_YEARS, fold=None)

	H5PATH = r'../data/H5dataset/kfold/'
	for k in range(nsplit):
		fold = json.load(open(r'../data/kfolds.json'))[f'fold_{k}']
		train_val, test = fold['train'], fold['test']

		buildH5Dataset(H5PATH, train_val, test, TRAIN_YEARS, TRAIN_YEARS, fold=k)

if __name__ == '__main__':
	main()
