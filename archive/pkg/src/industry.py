import clustering.make_clusters as cluster
import statarb.make_model_w_graphs as sa
from datetime import date
import pandas as pd
import numpy as np

clust = pd.read_csv('~/k_means/data/labled_stock_clusters.csv')
industry = pd.read_csv('~/k_means/data/constituents.csv')


keys = industry['Sector'].unique()
company_dict = dict.fromkeys(keys)

for i in keys:
	tickers = industry[industry['Sector'] == i ]['Symbol']
	company_dict[i] = list(tickers)

print(company_dict)