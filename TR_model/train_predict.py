'''
Module Description:
------
This module is used to train a time resolved model: LSTM (TR-LSTM) or Transformer network (TR-Transformer)

Author: 
------
Freddy Houndekindo (freddy.houndekindo@inrs.ca)

'''

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import shutil
from tqdm import tqdm
from model.Tranformer import TRTransformer
from model.LSTM import TRLSTM
import model_config

from dataloader import *

import argparse

# ====================================
#  Parser 
# ====================================

parser = argparse.ArgumentParser(description='Train model')

parser.add_argument('--model', type=str, default='LSTM', help='Model to train either LSTM or Transformer')
parser.add_argument('--data_path', type=str, default='../data/H5dataset/kfold/fold=0', help='H5dataset folder path')
parser.add_argument('--n_workers', type=int, default=1, help='number of workers for dataloader')
parser.add_argument('--in_memory', type=int, default=1, help = 'Set to 1 to load all data into memory or 0 to fetch batches from disk.')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs')

parser.add_argument('--ckpt_folder', type=str, default='./checkpoints', help='Folder to save model checkpoints')
parser.add_argument('--pred_folder', type=str, default='./pred', help='Folder to save the predictions')


args = parser.parse_args()


MCONFIG = {'LSTM':model_config.LSTM, 'Transformer':model_config.Transformer}[args.model]
NET = {'LSTM':TRLSTM, 'Transformer':TRTransformer}[args.model]
DATASET = {0:H5Dataset, 1:H5TensorDataset}[args.in_memory]
CKPTFOLDER = args.ckpt_folder
NWORKERS = args.n_workers
NEPOCHS = args.n_epochs
DATAPATH = os.path.normpath(args.data_path)

# Get the prediction folder and file name
if 'kfold' in DATAPATH:
	filename = f'{DATAPATH.split(os.sep)[-1]}.npy'
	folder = 'kfold'
if 'train_test' in DATAPATH:
	filename = 'test.npy'
	folder = 'train_test'

PREDFOLDER = f'{args.pred_folder}/{folder}'
os.makedirs(PREDFOLDER, exist_ok=True)
PREDPATH = f'{PREDFOLDER}/{filename}'

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

def training(config):

	"""
		Fonction to train the model
	
	"""

	trainSrcPath = f'{DATAPATH}/train.h5'
	valSrcPath = f'{DATAPATH}/val.h5'

	#============ Load dataset ============
	tDataset = DATASET(trainSrcPath)
	vDataset = DATASET(valSrcPath)

	trainLoader =  DataLoader(	tDataset,
								batch_size=config['batch_size'],
								num_workers=NWORKERS,
								shuffle=True,
								drop_last=True,
								pin_memory=True,
								pin_memory_device=device.type,
								persistent_workers= True if NWORKERS > 0 else False,

								)

	valLoader = DataLoader(	vDataset,
							batch_size=config['batch_size'],
							num_workers=NWORKERS,
							shuffle=False,
							drop_last=False,
							pin_memory=True,
							pin_memory_device=device.type,
							persistent_workers= True if NWORKERS > 0 else False,
							)
	
	#============ Create checkpoint folder ============
	folder = DATAPATH.split(os.sep)[-1] #either fold=x or train_test

	ckpPath = f'{CKPTFOLDER}/{folder}/'
	logPath = f'{ckpPath}log.csv'
	if shutil.os.path.exists(ckpPath):	shutil.rmtree(ckpPath)
	os.makedirs(ckpPath)

	with open(logPath, 'a') as log:
		log.write('lr,epoch,tloss,vloss,mpath\n')

	Model = NET(config).to(device)
	loss_fn = torch.nn.HuberLoss(reduction = 'none', delta=config['huber_slope']) 
	lr = config['lr']
	optimizer = torch.optim.Adam(Model.parameters(), lr=lr, weight_decay=0)

	print(f'Number of parameters in model: {NET.count_parameters(Model)}')

	def _validate():

		"""
			Function to validate
		
		"""

		vloss = 0
		with torch.no_grad():
			for _, batch in enumerate(tqdm(	valLoader,
								  			desc='validation',
											dynamic_ncols=True,
											ncols=10,
											leave=False)
											):

				inputs, target = prepare_batch(batch, device)


				outputs = Model(inputs)
				loss = loss_fn(outputs, target)
				loss = loss.mean()
				vloss += loss.item() / len(valLoader)

		return vloss

	def _training(epoch):

		"""
			Function to train an epoch
		
		"""

		trloss = 0
		for _, batch in enumerate(tqdm(	trainLoader,
								 		desc=f'training epoch={epoch}',
										dynamic_ncols=True,
										ncols=10,
										leave=False)
										):

			inputs, target = prepare_batch(batch, device)

			outputs = Model(inputs)

			loss = loss_fn(outputs, target)
			loss = loss.mean()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			trloss += loss.item() / len(trainLoader)
		return trloss


	for epoch in range(NEPOCHS):

		Model.train(True)
		trloss = _training(epoch)


		if epoch % 1 == 0:
			Model.eval()
			vloss = _validate()

			lr = optimizer.param_groups[0]['lr']
			print(f'lr: {lr} | Epoch: {epoch:02d} | Train Loss: {trloss:0.3f} | Val Loss: {vloss:0.3f}')

			mpath = f'{ckpPath}epoch={epoch}-vloss={vloss:0.3f}.pth'
			with open(logPath, 'a') as log:
				log.write(f'{lr},{epoch:02d},{trloss},{vloss},{mpath}\n')
			torch.save(Model.state_dict(), mpath)

	return logPath

def predict_test(config, logPath):

	"""
		Function to predict the test set
	
	"""

	testSrcPath = f'{DATAPATH}/test.h5'

	testDataset = DATASET(testSrcPath)

	testLoader =  DataLoader(	testDataset,
								batch_size=config['batch_size'],
								num_workers=NWORKERS,
								shuffle=False,
								drop_last=False,
							)

	

	log = pd.read_csv(logPath, index_col='epoch')
	ckpPath = log.loc[lambda x: x.vloss.idxmin(), :].mpath
	print(f'Test best model path: {ckpPath}')

	Model = NET(config)
	Model.load_state_dict(torch.load(ckpPath, weights_only=True))
	Model.to(device)


	Y_hat = []
	Model.eval()
	with torch.no_grad():
		for _, batch in enumerate(tqdm(testLoader, desc='test predict', dynamic_ncols=True, ncols=10, leave=False)):
			x, *_ = prepare_batch(batch, device)
			y_hat = Model(x)

			Y_hat += [y_hat.detach().cpu().numpy()]

	Y_hat = np.concatenate(Y_hat, axis=0)
	np.save(PREDPATH, Y_hat)

def main():

	#	Read the dataset configurations
	data_config = json.load(open(f'{DATAPATH}/config.json', 'r'))
	dconfig = {
				'plen':data_config['plen'],
				'flen':data_config['flen'],

				'staticFeatures':data_config['staticFeatures'],
				'eraFeatures':data_config['eraFeatures'],

				'static_key':'Xs',
				'dynamic_key':'Xd',
				'date_key': 'Xdate',

			}
	
	#	Merge the data and model configuration
	config = dconfig | MCONFIG
	#	Train the model
	logPath = training(config)
	#	predict the test set
	predict_test(config, logPath)


if __name__ == '__main__':
	main()