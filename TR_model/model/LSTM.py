import torch.nn as nn
import torch

# ====================================
#  Define the TR-LSTM model
# ====================================

class TRLSTM(nn.Module):
	def __init__(self, config):
		super(TRLSTM, self).__init__()

		# ============ Get configuration ============
		hidden_size = config['hidden_size']
		static_hidden_size = config['static_hidden_size']
		past_len = config['plen']
		future_len = config['flen']

		static_key = config['static_key']
		dynamic_key = config['dynamic_key']
		dates_key = config['date_key']

		dropout= config['dropout']
		dropout_static = config['dropout_static']
		num_layers = config['num_layers']

		nstatic = len(config['staticFeatures'])
		nera = len(config['eraFeatures'])
		ACTF = config['ACTF']

		# ============ Static features encoding ============
		self.MLP_static = nn.Sequential(
						nn.Linear(nstatic, static_hidden_size),
						getattr(nn, ACTF)(),
						nn.Dropout(dropout_static),
										)
		
		# ============ Dynamic features encoding ============
		self.lstm = nn.LSTM(    input_size=hidden_size,
					hidden_size=hidden_size,
					num_layers=num_layers,
					batch_first=True,
					bidirectional=False,
					dropout= dropout if num_layers > 1 else 0.0
								)
		self.dropout = nn.Dropout(dropout)
		self.temp_emb = TempEmbed(hidden_size)
		self.Xd_emb = nn.Linear(nera + nstatic, hidden_size)

		# ============ Output Linear mapping ============
		self.fc_out = nn.Linear(hidden_size + static_hidden_size, 1)

		self.past_len = past_len
		self.future_len = future_len
		self.seq_len = past_len + future_len
		self.static_key = static_key
		self.dynamic_key = dynamic_key
		self.dates_key = dates_key
		self.nera = nera

	def count_parameters(model): 
		return sum(p.numel() for p in model.parameters() if p.requires_grad)		

	def forward(self, x):

		# ============ Get features ============
		Xs =     x[self.static_key]
		Xd =      x[self.dynamic_key]
		Xdate = x[self.dates_key]

		# ============ Static features ============
		Xs_concat = Xs.unsqueeze(1).expand(-1, self.seq_len, -1)
		Xs = self.MLP_static(Xs)

		# ============-- dynamic feature encoding ============

		Xd = torch.cat([Xs_concat, Xd], dim=-1)
		temp_emb = self.temp_emb(Xdate)
		Xd = self.Xd_emb(Xd) + temp_emb

		Xd, _ = self.lstm(Xd)
		out = Xd[:, -self.future_len:, :]

		# ============ output projection ============
		out = self.dropout(out)
		Xs = Xs.unsqueeze(1).expand(-1, self.future_len, -1)
		out = torch.cat([out, Xs], dim=-1)
		out = self.fc_out(out).squeeze(-1)
		out = torch.nn.functional.softplus(out)
		return out
	
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
