# Statistical Arbitrage Using K-Means 

This project aims to use K-Means clustering to find non-traditional stock pairs to use in a stat arb trading model.

Main backetest notebook can be found in "testing" folder
- This test uses pairs from cluster 14 (randomly sellected) of the 31 total clusters created by the K-Means module ("sa_clusters.py")
- It contains 4 symbols each combination of which was backtested over a 24 mo. period.

To improve:

- Pair trade selection
  -   Hedge ratio currently uses OLS, implying no noise in observation of independent variable
  -   result means pair X/Y performs differently from Y/X
  -   Use TLS method to account for this?

- More robust feature selection:
  -   We want to choose fundamental statistics of stocks that we will use as features for clustering
  -   We need to account for things like autocorrelation and consider over/under specification
  -   e.g. not using gross profit and revenue  

- Re-Cluseter and split test every quarter
  -   As these numbers change per earnings period, the relationships between in-cluster pairs may have chnaged entirely

- Create overall portfolio metrics
  -   Currently we only test performance of one pair at a time
  -   In order to test over all performance of the model we must see performance aggregated accross pairs

- More accurate portfolio assumtions
  -   using adjusted close for entry price
  -   no account for margin needed to short or capacity of strategy overall 

