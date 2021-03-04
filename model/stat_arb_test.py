from stat_arb_clustering import Stat_Arb_Clustering as sac
from stat_arb_model import statarb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import adfuller
import yahoo_finance as yf
import pandas_datareader as pdr
from datetime import datetime
from datetime import date

random_state = 42

features_data = pd.read_csv('~/k_means/Data/sp_financials.csv')
features_data = features_data.drop(['SEC_Filings', 'Sector', 'Name'], axis=1)
symbol = features_data['Symbol']
features_data = features_data.set_index('Symbol')
features_data = features_data.fillna(0)
cols = ['Price', 'Price/Earnings', 'Dividend_Yield', 'Earnings/Share',
	   '52_Week_Low', '52_Week_High', 'Market_Cap', 'EBITDA', 'Price/Sales',
	   'Price/Book']
features_data = features_data[['Price', 'Price/Earnings', 'Dividend_Yield', 'Earnings/Share',
	   '52_Week_Low', '52_Week_High', 'Market_Cap', 'EBITDA', 'Price/Sales',
	   'Price/Book']]


# standardize data to have mean ~0 and var of 1
data_std = StandardScaler().fit_transform(features_data)

# save for later in df form
ds = pd.DataFrame(data_std, columns=cols)
ds = ds.set_index(symbol)

# create cluster object
# perform PCA on object
# find optimal K
# assign cluster label to data
# drop clusters with only one data point
# preview first cluster 
# get all combinations of pairs
clustered_data = sac(len(cols), 0.98, 42)
find_components = pd.DataFrame(clustered_data.get_pca(data = data_std))
yeet = clustered_data.get_clusters(data = find_components)
find_components['Clusters'] = yeet
find_components = find_components.set_index(symbol)
clean_data = clustered_data.drop_single_clusters(find_components)
cluster_1 = clean_data.loc[clean_data['Clusters'] == 1]
final_data = pd.DataFrame()
for i in range(max(yeet)):
	if len(clean_data.loc[clean_data['Clusters'] == i][1]) != 0:
		final_data = final_data.append(clean_data.loc[clean_data['Clusters'] == i])


user = ""
while user != "stop":

	C = int(input("Which cluster u wanna peep?: "))
	pairs = clustered_data.get_pairs(clean_data.loc[clean_data['Clusters'] == C])
	print(pairs)
	check = int(input("U sure u wanna see? (1=y, 0=n): "))
	if check == 1:
		number_of_pairs = len(pairs)
		start = datetime(2019, 2, 22)
		end = date.today()

		trade_data = []
		for i in range(number_of_pairs):
			p = pd.DataFrame()
			name_1 = pairs[i][0]
			name_2 = pairs[i][1]
			title = name_1 + " and " + name_2 + " Backtest"
			col_add = name_1 + "_" + name_2
			print(col_add)
			valid_pair = True
			try:
				s1 = pdr.get_data_yahoo(name_1, start, end)
			except:
				print("Can't get data for %s" %(name_1))
				valid_pair = False
			try:	
				s2 = pdr.get_data_yahoo(name_2, start, end)
			except:
				print("Can't get data for %s" %(name_2))
				valid_pair = False
			if valid_pair == True:
				stat_arb_object = statarb(s1, s2, name_1, name_2, 28, -2.0, 2.0, -3.0, 3.0, 14, start, end, 10000, title)
				spread = stat_arb_object.create_spread()
				stat_arb_object.check_cointergration()

				graphs = int(input("Wanna see graphs? (1=y, 0=n): "))
				if graphs == 1:
					signals = stat_arb_object.generate_signal()
					trades = stat_arb_object.generate_trades()
					portfolio = stat_arb_object.generate_portfolio()
					book = stat_arb_object.generate_daily_book()
					numbers = stat_arb_object.generate_metrics()

	user = input("Run again? (type stop to end): ")

