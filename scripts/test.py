from k_means import K_means

# import packages
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# define input arguments
n_samples = 1000
n_features = 2
n_clusters = 4
random_state = 25

# create initial blobs that we will base clusters around
X, y = make_blobs(n_samples = n_samples, n_features = n_features, centers = n_clusters, random_state = random_state)


model = K_means(n_clusters = n_clusters, random_state = random_state)
model.train(X)
init_centroids = model.init_centroids
centroids = model.fitted_model
print(centroids)

plt.plot(centroids[0][0], centroids[0][1], 'r*')
plt.plot(centroids[1][0], centroids[1][1], 'r*')
plt.plot(centroids[2][0], centroids[2][1], 'r*')
plt.plot(centroids[3][0], centroids[3][1], 'r*')
plt.plot(init_centroids[0][0], init_centroids[0][1], 'g*')
plt.plot(init_centroids[1][0], init_centroids[1][1], 'g*')
plt.plot(init_centroids[2][0], init_centroids[2][1], 'g*')
plt.plot(init_centroids[3][0], init_centroids[3][1], 'g*')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

'''
data = pd.read_csv('my_data.csv')
train, test = data[:1000], data[1000:]

model.train(train)

model.predict(test)
centroids = model.centroids
model.n_iter
'''

