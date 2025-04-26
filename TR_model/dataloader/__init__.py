import h5py
import torch
import math
import torch.nn as nn
from torch.utils.data import TensorDataset

def H5TensorDataset(srcPath):

	with h5py.File(srcPath, 'r') as dataset:
		Xd = dataset['DynamicFeatures'][:]
		target = dataset['target'][:]
		Xs = dataset['StaticFeatures'][:]
		Xdate = dataset['DateFeatures'][:]

	return TensorDataset(   torch.tensor(Xd, dtype=torch.float32),
                            torch.tensor(Xs, dtype=torch.float32),
					        torch.tensor(Xdate, dtype=torch.float32),
					        torch.tensor(target, dtype=torch.float32)
							)

class H5Dataset(torch.utils.data.Dataset):
	def __init__(self, srcPath):
		super().__init__()
		self.srcPath = srcPath
		self.dataset = None
		with h5py.File(self.srcPath, 'r') as f:
			self.len = f['DynamicFeatures'].shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, ix):

		if self.dataset is None:
			self.dataset = h5py.File(self.srcPath, 'r')

		Xd = self.dataset['DynamicFeatures'][ix]
		target = self.dataset['target'][ix]
		Xs = self.dataset['StaticFeatures'][ix]
		Xdate = self.dataset['DateFeatures'][ix]

		return (torch.tensor(Xd, dtype=torch.float32),
				torch.tensor(Xs, dtype=torch.float32),
				torch.tensor(Xdate, dtype=torch.float32),
				torch.tensor(target, dtype=torch.float32),
				)

def prepare_batch(batch, device='cuda'):
	Xd, Xs, Xdate, target = batch
	Xd = Xd.to(device)
	Xs = Xs.to(device)
	Xdate = Xdate.to(device)
	target = target.to(device)
	return {'Xd': Xd, 'Xs': Xs, 'Xdate': Xdate}, target

class TempEmbed(nn.Module):
	#https://github.com/thuml/Autoformer/blob/main/layers/Embed.py
	def __init__(self, hidden_size):
		super().__init__()
		self.m_emb = nn.Embedding(13, hidden_size)
		self.d_emb = nn.Embedding(32, hidden_size)
		self.h_emb = nn.Embedding(24, hidden_size)
	def forward(self, x):
		x = x.long()
		mx = self.m_emb(x[..., 0])
		dx = self.d_emb(x[..., 1])
		hx = self.h_emb(x[..., 2])
		return  hx + mx + dx