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

class Stat_Arb_Clustering(object):

	def __init__(self, n_components, variance, random_state):

		# should equal features in stock data frame
		self.n_components = n_components
		# desired explained variance cutoff for PCA
		self.variance = variance
		self.random_state = random_state

	def get_pca(self, data):

		# PCA instance
		# self.variance = 0.98
		self.pca = PCA(self.variance, n_components = self.n_components)
		self.principalComponents = self.pca.fit_transform(data)
		self.PCA_components = pd.DataFrame(self.principalComponents)

		# plot variances
		self.feat = range(self.pca.n_components_)
		plt.bar(self.feat, self.pca.explained_variance_ratio_)
		plt.xlabel('PCA Features')
		plt.ylabel('Variance %')
		plt.xticks(self.feat)
		plt.show()

		return self.principalComponents

	def get_clusters(self, data):

		self.data = data
		# different visualization
		self.model = KMeans(init='k-means++', random_state=self.random_state)
		self.visualizer = KElbowVisualizer(self.model, k=(2,150))
		self.visualizer.fit(self.data)
		self.visualizer.show()

		self.K = self.visualizer.elbow_value_
		# implement model with K found above
		self.kmeans = KMeans(n_clusters = self.K, init='k-means++', random_state = self.random_state)
		self.label = self.kmeans.fit_predict(self.data)
		self.u_labels = np.unique(self.label)

		return self.label

	def drop_single_clusters(self, data):

		self.data = data
		self.pairs = pd.DataFrame()
		self.pairs = pd.concat(i for self.pairs, i in self.data.groupby(by=self.data['Clusters']) if len(i)>1)

		return self.pairs 

	def get_pairs(self, data):

		self.data = data
		self.X = self.data.index
		self.Y = self.data.index
		self.X_sqr = [[x, y] for x in self.X for y in self.Y]
		self.pairs = []
		self.ref = 0
		for i in range(len(self.X)):
		    for j in range(self.ref+1, int(np.sqrt(len(self.X_sqr)))):
		        self.pairs.append(self.X_sqr[j+(self.ref * int(np.sqrt(len(self.X_sqr))))])
		    self.ref += 1
		####### need to work on vectorizing operations using np.arrays ########
		# can use np.triu for above opperation
		# update in future
		return self.pairs





