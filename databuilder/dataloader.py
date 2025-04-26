import pandas as pd
import numpy as np


class Dataloader():

	def __init__(self, config):
		
		PATHS = config['Paths']
		self.years = config['years']
		self.Climate_ID = config['Climate_ID']
		self.train_Climate_ID = config['train_Climate_ID']
		self.obs_col = config['obs_col']
		self.era_col = config['era_col']
		self.eraPath = PATHS['eraPath']
		self.StaticPath = PATHS['StaticPath']

		self.obsPath = PATHS['obsPath']
		self.eraFeatures = config['eraFeatures']
		self.staticFeatures = config['staticFeatures']
		
		self.plen = config['plen']
		self.flen = config['flen']

		self.readMeta()

	def __len__(self):
		return len(self.batch)
						
	def readMeta(self):
		filters = [('Year', 'in', self.years), ('Climate_ID', 'in', self.Climate_ID)]
		self.batch = (pd.read_parquet(self.obsPath, filters=filters, columns=['Date'])
							.reset_index()
							.drop(columns='Date_UTC')
							.drop_duplicates(subset=['Climate_ID', 'Date'])
							.set_index(['Date', 'Climate_ID'])
							.sort_index()
							.assign(Date00h00 = lambda x: (pd.to_datetime(x.index.get_level_values('Date'), format='%Y-%m-%d %H:%M')) )
							)
		self.Static_data = pd.read_csv(self.StaticPath, index_col='Climate_ID')
		
	def build_dates(self):
		if self.plen > 0:
			past = (pd.concat(
						[self.batch.assign(temp = lambda x: x['Date00h00'] - pd.Timedelta(hours=i))
								.rename(columns={'temp':f'P-{i}'})
								.drop(columns='Date00h00') 
							for i in range(1, self.plen+1)
						], 
					axis=1)
					)
		else:
			past = pd.DataFrame()
		
		future = (pd.concat(
							[self.batch.assign(temp = lambda x: x['Date00h00'] + pd.Timedelta(hours=i))
									.rename(columns={'temp':f'f-{i}'})
									.drop(columns='Date00h00') 
								for i in range(0, self.flen)
							], 
						axis=1)
						)
		
		self.fdates = future
		self.dates_long = (pd.concat([past, future], axis=1)
								.melt(ignore_index=False, var_name='Type', value_name='Date_UTC')
								.sort_index()
								.reset_index()
								.set_index(['Date', 'Climate_ID', 'Date_UTC'])
								.sort_index()
				
							)
		
	def load_dates(self):

		Index = pd.MultiIndex.from_frame(self.dates_long.reset_index()[['Climate_ID', 'Date_UTC']])
		years = Index.get_level_values('Date_UTC').year
		
		filters = [('Year', 'in', years), ('Climate_ID', 'in', self.Climate_ID)]
		
		Xdates = (	pd.read_parquet(
									self.eraPath, 
									filters=filters,
									columns=['Date_local']
									)
						.reset_index()
						.set_index(['Climate_ID', 'Date_UTC'])
						.loc[Index, ['Date_local']]
						.assign(Hour = lambda x: pd.DatetimeIndex(x.Date_local).hour,
						 		Month = lambda x: pd.DatetimeIndex(x.Date_local).month,
								day = lambda x: pd.DatetimeIndex(x.Date_local).day)
						.loc[:, ['Month', 'day', 'Hour']]
						.values
						.reshape(-1, self.plen + self.flen, 3)
						)

		
		return Xdates

	def load_target(self):

		assert self.flen == 24, 'flen length equal 24'
		
		Index = (self.fdates
					.melt(ignore_index=False, var_name='Type', value_name='Date_UTC')
					.reset_index()
					.set_index(['Date', 'Climate_ID', 'Date_UTC'])
					.sort_index()
					.index
				)
	
		filters = [('Year', 'in', self.years), ('Climate_ID', 'in', self.Climate_ID)]
		
		target = (	pd.read_parquet(
									self.obsPath, 
									filters=filters,
									columns=['Date', self.obs_col, self.era_col] 
									)
						.assign(target = lambda x: x[self.obs_col] / x[self.era_col])
						.reset_index()
						.set_index(['Date', 'Climate_ID', 'Date_UTC'])
						.loc[Index, ['target']]
						.values
						.reshape(-1, self.flen)
						)
		return target

	def load_static(self):

		Xs = self.Static_data.loc[self.batch.index.get_level_values('Climate_ID'), self.staticFeatures].values
		strain = self.Static_data.loc[self.train_Climate_ID, self.staticFeatures].values
		smin = strain.min(axis=0).reshape(1, -1)
		smax = strain.max(axis=0).reshape(1, -1)
		Xs = (Xs - smin) / (smax - smin)
		return Xs
	
	def load_era(self):
		
		Index = pd.MultiIndex.from_frame(self.dates_long.reset_index()[['Climate_ID', 'Date_UTC']])
		Years = Index.get_level_values('Date_UTC').year.unique()

		filters = [('Year', 'in', Years), ('Climate_ID', 'in', self.Climate_ID)]
		
		Xd = []
		for feature in self.eraFeatures:
			if  feature in ['10u', '10v']:
				xd  = pd.read_parquet(	self.eraPath, #multiindex = 'Climate_ID', 'Date_UTC'
									columns = [feature, '10ws'], 
									filters = filters			
								)
				xd = xd.assign(temp = lambda x: x[feature]/(x['10ws'] + 1e-5)).rename(columns={'temp':f'n{feature}'})
				Xd += [xd.loc[Index, [f'n{feature}']].values]
			else:
				xd  = pd.read_parquet(	self.eraPath, #multiindex = 'Climate_ID', 'Date_UTC'
									columns = [feature], 
									filters = filters			
								)
				xd = (xd.assign(temp = lambda x: self.Static_data.loc[x.index.get_level_values('Climate_ID'), f'{feature}_mean'].values)
		  				.rename(columns={'temp':f'{feature}_mean'})
						.assign(temp = lambda x: self.Static_data.loc[x.index.get_level_values('Climate_ID'), f'{feature}_std'].values)
		  				.rename(columns={'temp':f'{feature}_std'})
						.assign(temp = lambda x: (x[feature] - x[f'{feature}_mean'])/x[f'{feature}_std'] )
						.rename(columns={'temp':f'n{feature}'})
						
						)
				Xd += [xd.loc[Index, [f'n{feature}']].values]
				
			

		return np.stack(Xd, -1).reshape(-1, self.plen + self.flen,  len(self.eraFeatures))

