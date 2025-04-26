LSTM = {
				'dropout':0.2,
				'dropout_static':0.05,
				'hidden_size':256,
				'static_hidden_size':32,

				'num_layers': 1,
				'ACTF': 'GELU',
				'huber_slope':1.5,

				'lr':2e-5,
				'batch_size': 128,
			}

Transformer = {
				'dropout':0.1,
				'dropout_static':0.05,
				'hidden_size':256,  
				'static_hidden_size':32,

				'num_layers': 1,
				'ACTF': 'GELU',
				'huber_slope':1.5,

				'lr':1e-5,
				'batch_size': 128,
                'nblock':4,
                'nheads':4,
                
			}