# Statistical Arbitrage Using K-Means 

This project aims to use K-Means clustering in a stat arb trading model to find non-traditional pairs.

Main backetest notebook can be found in "testing" folder
- This test uses pairs from cluster 14 (randomly sellected) of the 31 total clusters created by the K-Means module ("sa_clusters.py")
- It contains 4 symbols each combination of which was backtested over a 24 mo. period.
- Some pairs performed well others did not. 
- Two stand out examples were BAC/WFC & C/JPM, demonstraing high volatilty.

Next Steps:
- Reduce idiosyncratic exposure by conducting rolling backtests in order to better select pairs (i.e. don't trade every pair per cluster)
-   build in X/Y vs Y/X test as rolling regression is not orthognal?
- write method to find cumulative returns of all pairs withing clusters
- sum these returns accross clusters
