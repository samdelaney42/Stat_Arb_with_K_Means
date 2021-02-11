# K_Means

In this project I aim to re-create the unsupervised machine learning k-means clustering technique with random and plusplus initialization methods, then use class in a stat arb trading model to find non-traditional pairs.

To do:
- Implement in statistical arbitrage pair trading model
- Test stat arb strat with different:
  - beta lookback periods
  - Z score entry and exit thresholds
- Itterate through clusters and generate porfolios with multiple pairs positions
- explore using batch API calls to retrive stock data

- implement stop loss function at spread z-score = 3 sigma
  - could pair this with re performing ADF to check if basic hypothesis has been nullified 
  
- Take profit function at mean reversion to 0? depends on risk apetite.  
