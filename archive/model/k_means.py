import numpy as np

class K_means(object):
	'''
	k-means object with ..... atributes 
	methods
	-------
	train:

	predict:
	inputs: data: np.array, n x d

	attributes
	-----
	'''
	def __init__(self, n_clusters, init_method = 'k++', random_state = 1):
		self.n_clusters = n_clusters
		self.init_method = init_method
		self.centroids = None


	def train(self, data):
		'''
			what should data look like?
		'''
		# initialize
		if self.init_method == 'k++':
			self.init_centroids = self.plus_plus(data, self.n_clusters, random_state = 1)
		elif self.init_method == 'random':
			self.init_centroids = self.random(data, self.n_clusters, random_state = 1)
			###
		# Train centroids on data
			###
		self.fitted_model = self.fit_centroids(data, self.n_clusters, self.init_centroids)
		#self.fit_centroids(data) #make this method and place n_iter here

	def predict(self, data):
		print("yeah")

	def plus_plus(self, X, n_clusters, random_state):
		'''
		parameters
		----------
			X: numpy.ndarray: All points created by make blobs
			n_clusters: int: number of clusters we want (= centers parameter in make_blobs())
			random_state: int: desired seed for repeatable results
		returns
		----------
			initial_centroid_points: list: ++ chosen points representing our initial centroids 
		'''
		# set seed for testing repeatability
		np.random.seed(random_state)
		# choose single random centroid from X
		rc = np.random.choice(len(X)-1)
		[x, y] = X[rc, 0], X[rc, 1]
		rand_centroid = np.array([x, y])
		# prepare output array of centroids begining with randomly selected first point
		centroids = [rand_centroid]
		# iterate through desired number of clusters
		for i in range(1, n_clusters):
			distance_square = []
			# iterate through all points
			for j in X:
				point_dist_square = []
				# find square distance between current point and given centroid
				for k in centroids:
					# np.inner returns square distsnce
					point_dist_square.append([np.inner(k-j, k-j)])
				# find which centroid is closest to current point with min square dist
				distance_square.append(min(point_dist_square))
			distance_square = np.array(distance_square)
			# find probabilities of point being next centroid candidate
			probability = distance_square/distance_square.sum()
			# Select the next centroid by choosing point with highest probability 
			cumulative_probabilities = probability.cumsum()
			rand = np.random.rand()
			for j, p in enumerate(cumulative_probabilities):
				if rand < p:
					i = j
					print(i)
					print(X[i])
					break
			centroids.append(X[i])
		return centroids

	def random(self, X, n_clusters, random_state):
		'''
		parameters
		----------
			X: numpy.ndarray: All points created by make blobs
			n_clusters: int: number of clusters we want (= centers parameter in make_blobs())
			random_state: int: desired seed for repeatable results
		returns
		----------
			initial_centroid_points: list: randomly chosen points representing our initial centroids 
		'''
		# choose random numbers between 0 and size of X to use as index to pull random point coordinates
		# choose as many random numbers as equal to number of clusters
		rand_centroids = np.random.choice(len(X)-1, n_clusters, replace = False)
		# set up output list
		initial_centroid_points = []
		# use random number as index to pull point coordinates 
		for i in rand_centroids:
			[x, y] = X[i, 0], X[i, 1]
			initial_centroid_points.append([x, y])
		# return list of randomly selected coordinates
		return initial_centroid_points

	def cluster_points(self, X, n_clusters, initial_centroid_points):
		'''
		parameters
		----------
			X: numpy.ndarray: of all points created by make blobs
			n_clusters: int: number of clusters we want (= number of centers parameter in make_blobs())
			initial_centroid_points: list: of randomly chosen initial centroids 
		returns
		----------
			clusters: numpy.ndarray: of clusters. Points grouped by min distnace to closest centroid 
		'''
		# get number of desired clusters
		p = n_clusters
		# set up output list
		clusters = []
		# nest empty lists in output list based on number of clusters desired
		for n in range(p):
			points = []
			clusters.append(points)
		# iterate over every point in data set pulling x and y coordinates
		for x in X:
			x1 = x[0]
			y1 = x[1]
			# list for distances between current point and each centroid
			dists = []
			# itterate over each initial centroid to find distances to current point
			for c in initial_centroid_points:
				x2 = c[0]
				y2 = c[1]
				# find distance between current point and centroid
				d = self.distance_calc(x1, y1, x2, y2)
				dists.append(d)
			# find the index in distance list of minimum distance 
			# use this index to place current point in correct nested list in output list 
			index = dists.index(min(dists))
			clusters[index].append(x)
		# return clusters
		return np.array(clusters, dtype=object)

	# distance calculator method using euclidian method
	def distance_calc(self, x1, y1, ix, iy):
		'''
		parameters
		----------
			x1, y1: numpy.float64: centroid coordinates
			ix, yx: numpy.float64: point in blob coordinates
		returns
		----------
			distance: numpy.float64: Euclidian distance between centroid and point in blob
		'''
		d = np.sqrt(np.square(x1 - ix) + np.square(y1 - iy))
		return d

	# find true centroid 
	def true_centroid(self, clusters):
		'''
		parameters
		----------
			clusters: nimpy.ndarray: collection of points in cluster
		returns
		----------
			mean distance: list: mean x and y position of points in cluster
		'''
		new_centroid_points = []
		for i in clusters:
		# take x and y coordinates of each point in cluseter
			x = []
			y = []
			for j in i:
				x.append(j[0])
				y.append(j[1])
			# find mean position of these points
			x_mean = np.mean(x[:])
			y_mean = np.mean(y[:])
			# return as point that we will use as new centroid
			points = [x_mean, y_mean]
			new_centroid_points.append(points)
		# output 
		return new_centroid_points

	def fit_centroids(self, data, n_clusters, centroids):
		'''
		parameters
		----------
			data: numpy.ndarray: of all points created by make blobs
			n_clusters: int: number of clusters we want (= number of centers parameter in make_blobs())
			centroids : list: initial centroid points
		returns
			new_centroid_points: list: optimized centroid points
		----------
		'''
		# make range n_iter
		for x in range(500):
			itter_centroid_points = []
			# find new clusters with newly found centroid
			clusters = self.cluster_points(data, n_clusters, centroids)
			# find new centroid based on new clusters with true_centroid() method
			itter_centroid_points = self.true_centroid(clusters)
			new_centroid_points = itter_centroid_points
		return(new_centroid_points)




