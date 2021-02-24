# K_Means

In this project I aim to re-create the unsupervised machine learning k-means clustering technique with random and plusplus initialization methods, then use class in a stat arb trading model to find non-traditional pairs.

To do:
- Test stat arb strat with different:
  - beta lookback periods
  - Z score entry and exit thresholds
- Itterate through clusters and generate porfolios with multiple pairs positions
- explore using batch API calls to retrive stock data

- implement stop loss function at spread z-score = 3 sigma
  - could pair this with re performing ADF to check if basic hypothesis has been nullified 
  
- PCA for features?
- trailing stop loss
  - Add discrete positon sizing
  - Add $ value stop loss / take profit 
- daily updated portfolio summary
- IEX & EDGAR API Daily pull to keep base clustering info up to date 
- performance metrics:
  - CAGR
  - Sharpe
  - Calmar
  - Maximum Drawdown
  - Maximum Drawdown Duration

  - Sharpe assumes Gaussian returns. That is why Maximum Drawdown is used to reveal the tail risks. Similar assumptions when you compute the portfolio risk & return. Beware of the covariance matrix! 

- Dynamic time warping for some features used in kmeans?
   
