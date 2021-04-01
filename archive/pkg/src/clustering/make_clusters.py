import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter

import seaborn as sns
from statsmodels.tsa.api import adfuller
import yahoo_finance as yf
import pandas_datareader as pdr

# Modify some settings
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 150

pd.options.display.max_rows = 20
pd.options.display.max_columns = 15

class clusters(object):
	'''
	Attributes
	----------
		path: string
			path to data set containing features of stocks on S&P 500
		variance: float
			desired explained variance threshold for PCA (default = 0.98)
		random_state: int
			random state for repeatable testing with centroid initialization (default = 42)
		show_plot: boolean
			show PCA variance plots (default = True)
	----------
	
	Methods
	-------
		create()
			runs methods required to generate clusters of stocks
		generate_std_data()
			standardizes input data
		generate_pca()
			runs PCA on input data
		generate_clusters()
			finds elbow point for k-means and applies this value, generating clusters of stocks
		generate_low_num_clusters()
			finds clusters containg less than the desired number of stocks. used for ease of selecting pairs
		generate_pairs()
			finds all combinations of pairs of stocks in a given cluster
	-------
	'''

	def __init__(self, path, variance = 0.98, random_state = 42, show_plot = True):
		
		self.data = pd.DataFrame()
		self.path = path
		self.variance = variance
		self.random_state = random_state
		self.show_plot = show_plot
		
	def create(self):
		'''
		parameters
		----------
			self: object 
				clusters object
		returns
		----------
			nothing
				gets, standardises data. performs pca, optimizes for k and runs k-means
		'''
		
		self.get_std_data = self.generate_std_data()
		self.get_pca = self.generate_pca()
		self.get_clusters = self.generate_clusters()
		
	def generate_std_data(self):
		'''
		parameters
		----------
			self: object 
				clusters object
		returns
		----------
			nothing
				standardizes input data
		'''
		
		# pull in data
		self.data = pd.read_csv(self.path)
		# drop unnescesary cols
		self.data = self.data.drop(['SEC_Filings', 'Sector', 'Name'], axis=1)
		# save symbols for late
		self.symbol = self.data['Symbol']
		# set index as symbols
		self.data = self.data.set_index('Symbol')
		self.data = self.data.fillna(0)
		# rename cols
		self.cols = ['Price', 'Price/Earnings', 'Dividend_Yield', 'Earnings/Share',
			   '52_Week_Low', '52_Week_High', 'Market_Cap', 'EBITDA', 'Price/Sales',
			   'Price/Book']
		self.data = self.data[['Price', 'Price/Earnings', 'Dividend_Yield', 'Earnings/Share',
			   '52_Week_Low', '52_Week_High', 'Market_Cap', 'EBITDA', 'Price/Sales',
			   'Price/Book']]
		
		# standardize data
		self.data_std = StandardScaler().fit_transform(self.data)
		return self.data_std
	
	def generate_pca(self):
		'''
		parameters
		----------
			self: object 
				clusters object
		returns
		----------
			PCA_components: df
				dataframe, index = stock tickers, cols = selected standardised data from pca
		'''
		# pca model
		self.pca = PCA(n_components = len(self.data_std[0]))
		self.principalComponents = self.pca.fit_transform(self.data_std)
		self.PCA_components = pd.DataFrame(self.principalComponents)
		self.PCA_components = self.PCA_components.set_index(self.symbol)

		# plot variances
		if self.show_plot == False:
			print("show_plot = False")
		else:
			with plt.style.context(['seaborn-paper']):
				self.feat = range(self.pca.n_components_)
				plt.bar(self.feat, self.pca.explained_variance_ratio_)
				plt.xlabel('PCA Features')
				plt.ylabel('Variance %')
				plt.xticks(self.feat)
				plt.show()

		return self.PCA_components
	
	def generate_clusters(self):
		'''
		parameters
		----------
			self: object 
				clusters object
		returns
		----------
			two_or_more: df
				index = stock tickers, cols = selected standardised data from pca and cluster labels
				two_or_more means we have removed clusters that contain only one stock
		'''

		self.model = KMeans(init='k-means++', random_state=self.random_state)
		self.visualizer = KElbowVisualizer(self.model, k=(2,150))
		self.visualizer.fit(self.PCA_components)
		self.visualizer.show()

		self.K = self.visualizer.elbow_value_
		# implement model with K found above
		self.kmeans = KMeans(n_clusters = self.K, init='k-means++', random_state = self.random_state)
		self.label = self.kmeans.fit_predict(self.PCA_components)
		self.u_labels = np.unique(self.label)
		self.PCA_components['Clusters'] = self.label
		self.two_or_more = pd.concat(i for self.two_or_more, i in self.PCA_components.groupby(by=self.PCA_components['Clusters']) if len(i)>1)

		self.two_or_more.to_csv('~/k_means/data/labled_stock_clusters.csv')

		return self.two_or_more
	
	def generate_low_num_clusters(self, max_num_stocks):
		'''
		parameters
		----------
			self: object 
				clusters object
			max_num_stocks: int
				max number of stocks desired per cluster
		returns
		----------
			wanted_clusters: list
				containing the labels of clusters with 0 < num_stocks <= max_num_stocks
		'''
		
		self.max_num_stocks = max_num_stocks
		self.wanted_clusters = []
		
		for i in range(self.K):
			self.num_stocks = len(self.two_or_more[self.two_or_more['Clusters'] == i])
			if self.num_stocks == 0:
				continue
			elif self.num_stocks <= self.max_num_stocks:
				self.wanted_clusters.append(i)
		
		print("Clusters with <= {} stocks: ".format(self.max_num_stocks), self.wanted_clusters)
		
		return self.wanted_clusters
	
	def generate_pairs(self, which_cluster):
		'''
		parameters
		----------
			self: object 
				clusters object
			which_cluster: int
				desired cluster containing stocks that you want to find all possible combinations of
		returns
		----------
			pair_combinations: list[list]
				nested lists with all possible unique combinations of pairs
		'''
		
		self.which_cluster = which_cluster
		self.data = self.two_or_more[self.two_or_more['Clusters'] == self.which_cluster]
		self.X = self.data.index
		self.Y = self.data.index
		self.X_sqr = [[x, y] for x in self.X for y in self.Y]
		self.pair_combinations = []
		self.ref = 0
		for i in range(len(self.X)):
			for j in range(self.ref+1, int(np.sqrt(len(self.X_sqr)))):
				self.pair_combinations.append(self.X_sqr[j+(self.ref * int(np.sqrt(len(self.X_sqr))))])
			self.ref += 1

		print("Unique stock combinations in cluster {}: ".format(self.which_cluster), self.pair_combinations)
		####### need to work on vectorizing operations using np.arrays ########
		# can use np.triu for above opperation
		# update in future
		return self.pair_combinations