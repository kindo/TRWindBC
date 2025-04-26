import math
import torch.nn as nn
import torch


# ====================================
#  Define the TR-Transformer model
# ====================================

class TRTransformer(nn.Module):
	def __init__(self, config):
		super(TRTransformer, self).__init__()

		# ============ Get configuration ============
		hidden_size = config['hidden_size']
		static_hidden_size = config['static_hidden_size']
		past_len = config['plen']
		future_len = config['flen']

		static_key = config['static_key']
		dynamic_key = config['dynamic_key']
		dates_key = config['date_key']

		dropout = config['dropout']
		dropout_static = config['dropout_static']
		nblock = config['nblock']
		nheads = config['nheads']

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
		self.encoder = nn.Sequential(*[GPT2Block(
														hidden_size=hidden_size,
														nheads=nheads,
														dropout=dropout,
														ACTF=ACTF
												)

												for _ in range(nblock)])
		self.dropout = nn.Dropout(dropout)
		self.temp_emb = TempEmbed(hidden_size)
		self.pos_emb = PositionalEmbedding(hidden_size)
		self.Xd_emb = nn.Linear(nera + nstatic, hidden_size)
		self.fc_out = nn.Linear(hidden_size + static_hidden_size, 1)
		self.ln_out = nn.LayerNorm(hidden_size)

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

		# ============ Static features encoding ============
		Xs_concat = Xs.unsqueeze(1).expand(-1, self.seq_len, -1)
		Xs = self.MLP_static(Xs)

		# ============-- dynamic feature encoding ============
		Xd = torch.cat([Xs_concat, Xd], dim=-1)
		temp_emb = self.temp_emb(Xdate)
		pos_emb = self.pos_emb(Xd)
		Xd = self.Xd_emb(Xd) + temp_emb + pos_emb

		Xd = self.encoder(Xd)
		Xd = self.ln_out(Xd)
		out = Xd[:, -self.future_len:, :]
            
		# ============ output projection ============
		Xs = Xs.unsqueeze(1).expand(-1, self.future_len, -1)
		out = torch.cat([out, Xs], dim=-1)
		out = self.fc_out(out).squeeze(-1)
		out = torch.nn.functional.softplus(out)

		return out


class PositionalEmbedding(nn.Module):
    #https://github.com/thuml/Autoformer/blob/main/layers/Embed.py#L84
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

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


class GPT2Block(nn.Module):
    #https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    def __init__(   self,
                    hidden_size,
                    nheads=4,
                    dropout=0.1,
                    ACTF='GELU'
                 ):

        super().__init__()

        self.mlp = nn.Sequential(   nn.Linear(hidden_size, 4*hidden_size), 
                                    getattr(nn, ACTF)(),
                                    nn.Linear(4*hidden_size, hidden_size),
                                    nn.Dropout(dropout)
                                    )

        self.mattn = MHAPyTorchScaledDotProduct(hidden_size=hidden_size,
                                                num_heads=nheads,
                                                dropout=0,
                                                )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):

        x = x + self.mattn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MHAPyTorchScaledDotProduct(nn.Module):
#https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
    def __init__(self, hidden_size, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)

        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # (b, seq_len, hidden_size) --> (b, seq_len, 3 * hidden_size)
        qkv = self.qkv(x)

        # (b, seq_len, 3 * hidden_size) --> (b, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # (b, seq_len, 3, num_heads, head_dim) --> (3, b, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, seq_len, head_dim) -> 3 times (b, num_heads, seq_len, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout
        out = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where hidden_size = self.num_heads * self.head_dim
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        return out
