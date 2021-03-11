import clustering.make_clusters as cluster
import statarb.make_model_w_graphs as sa
from datetime import date
import pandas as pd
import numpy as np

# get clusters
path = '~/k_means/Data/sp_financials.csv'
c = cluster.clusters(path)
c.create()
low_num_clusters = c.generate_low_num_clusters(max_num_stocks = 10)

user = ""
while user != "stop":

	select_cluster = int(input("Select a cluster of stocks from the list above (I recomend 11 mitchell): "))
	selected_cluster = c.generate_pairs(which_cluster = select_cluster)

	in_cluster = ""
	while in_cluster != 0:
		select_pair = int(input("Select index of pair combination to backtest from the list above (0 to {}): ".format(len(selected_cluster)-1)))

		start = date(2016, 1, 1)
		end = date.today()
		name_1 = selected_cluster[select_pair][0]
		name_2 = selected_cluster[select_pair][1]

		print("Backtesting {} and {} from {} to {} ......".format(name_1, name_2, start, end))

		valid_pair = sa.StatArbBuilder(name_1, name_2, start, end).build()

		if valid_pair == True:
			model = sa.StatArb(name_1, name_2, start, end)
			model.backtest_pair()
		else:
			print("Cannot backtest pair")

		in_cluster = int(input("View another pair in this cluster? (1=y, 0=n): "))

	user = input("View another cluster or end? (type stop to end): ")

	if user != "stop":
		print(low_num_clusters)