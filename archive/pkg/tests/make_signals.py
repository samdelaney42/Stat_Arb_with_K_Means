import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import yahoo_finance as yf
import pandas_datareader as pdr
from datetime import date
from sklearn.cluster import KMeans
from statsmodels.tsa.api import adfuller
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from statsmodels.regression.rolling import RollingOLS
import plotly.express as px
import plotly.graph_objects as go

# Modify some settings
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 150

pd.options.display.max_rows = 20
pd.options.display.max_columns = 15

class statarb(object):
    '''
    Attributes
    ----------
        s1: str
            name of stock 1
        s2: str
            name of stock 2
        start: datetime
            start date of backtest period
        end: datetime
            end date of backtest period
        ma: int
            moving average window used in calc of rolling z-score (default = 28)
        floor / ceiling: float
            trade entry criteria (default = +/-2.0)
        stop_loss_long / short: float
            trade stop loss criteria (default = +/-3.0)
        beta_lookback: int
            lookback window used when calculating hedge ratio (default = 14)
        allocation: int
            dollar value of funds to use (default = 10,000)
        exit_zscore: float
            Z-score at which we close position (default = 0) 
        show_plot: boolean
            define if you want to see charts
        hist_adf: boolean
            decide if you want to backtest conditional on ADF result. i.e. only trade when coint
    ----------
    
    Methods
    -------
        show_pair()
            displays daily price charts of chosen stocks through chosen dates
        test_coint()
            returns if pair is cointergrated to quickly check if its worth back testing
        backtest_pair()
            backtests pair using all methods below
        generate_spread()
            takes daily closing prices, caluclating the hedge ratios and spreads of the assets
        generate_cointergration()
            Checks if the pair is currently cointergrated and to what significance
        generate_signal()
            takes spread and generates signals indicating trade entry / exit / stop out for long and short positions
        generate_trades()
            defines the postions of both stocks in the pair. i.e. long trade, stock 1 = positive, stock 2 = negative
        generate_portfolio()
            shows our portfolio returns and equity curve
        generate_order_book()
            shows the dates on which we would of placed orders (buy or sell) for each stock and quantity of shares for each
        generate_metrics()
            Shows various portfolio and return metrics e.g. sharp ratio, avg wins / lossesetc
    -------
    '''
    def __init__(self, s1, s2, start, end, ma=28, floor=-2.0, ceiling=2.0, stop_loss_long=-3.0, stop_loss_short=3.0, beta_lookback=28, allocation=10000, exit_zscore=0, show_plot = True, hist_adf = False):
        self.s1 = s1 # name of stock one
        self.s2 = s2 # name of stock two 
        self.df1 = pdr.get_data_yahoo(s1, start, end) # dataframe of stock one
        self.df2 = pdr.get_data_yahoo(s2, start, end) # dataframe of stock two
        self.df = pd.DataFrame(index = self.df1.index) # new df for data_cleaning method
        self.signals = pd.DataFrame(index = self.df1.index)
        self.trades = pd.DataFrame(index = self.df1.index)
        self.portfolio = pd.DataFrame(index = self.df1.index)
        self.metrics = pd.DataFrame(index = self.df1.index)
        self.book = pd.DataFrame(index = range(len(self.df1)))
        self.ma = ma # moving average period
        self.floor = floor # buy threshold for z-score
        self.ceiling = ceiling # sell threshold for z-score
        self.stop_loss_long = stop_loss_long # z-score continues to drop through our floor
        self.stop_loss_short = stop_loss_short # z-score continues to rise through our ceiling 
        self.Close = 'Close Long'
        self.Cover = 'Cover Short'
        self.beta_lookback = beta_lookback # lookback of beta for hedge ratio
        self.start = start # begining of test period
        self.end = end # end of test period
        self.exit_zscore = exit_zscore # z-score at which trade is closed
        self.allocation = allocation # dollar value of funds
        self.pair = self.s1 + " and " + self.s2 + " Backtest"
        self.show_plot = show_plot 
        self.hist_adf = hist_adf
        

            
    def show_pair(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
		'''
        if self.show_plot == False:
            print("show_plot = False")
        else:
            with plt.style.context(['seaborn-paper']): 
                ma1 = self.df1['Close'].rolling(window=self.ma).mean()
                std1 = self.df1['Close'].rolling(window=self.ma).std() 
                upper1 = ma1 + (std1 * 2)
                lower1 = ma1 - (std1 * 2)

                ma2 = self.df2['Close'].rolling(window=self.ma).mean()
                std2 = self.df2['Close'].rolling(window=self.ma).std() 
                upper2 = ma2 + (std2 * 2)
                lower2 = ma2 - (std2 * 2)

                plt.plot(self.df1['Close'],label=self.s1)
                plt.plot(upper1, 'r', alpha=0.5)
                plt.plot(lower1, 'r', alpha=0.5)
                plt.plot(ma1, 'r', alpha=0.5)
                plt.fill_between(self.df1.index, upper1, lower1, alpha=0.1)
                plt.plot(self.df2['Close'],label=self.s2)
                plt.plot(upper2, 'g', alpha=0.5)
                plt.plot(lower2, 'g', alpha=0.5)
                plt.plot(ma2, 'g', alpha=0.5)
                plt.fill_between(self.df1.index, upper2, lower2, alpha=0.1)
                plt.title(self.s1 + ' and ' + self.s2 + ' ' + str(self.start) + ' to ' + str(self.end))
                plt.legend(loc=0)
                plt.show()
                
    def test_coint(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
            nothing
                a check of cointergration. Useful for running through combinations of stocks in a cluster
		'''
        
        self.generate_spread()
        self.generate_cointergration()
            
    def backtest_pair(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
            nothing
                fully backtests a given pair of stocks
		'''
        
        self.get_spread = self.generate_spread()
        self.get_coint = self.generate_cointergration()
        self.get_signals = self.generate_signal()
        self.get_trades = self.generate_trades()
        self.get_portfolio = self.generate_portfolio()
        self.get_book = self.generate_order_book()
        self.get_metrics = self.generate_metrics()
        self.get_ts = self.trade_summary()
        
        
        
    def generate_spread(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			self: df 
                dataframe with chosen stock prices cov, var, beta, hedge ratio and spreads
		'''
        
        # take closing price of chose stocks and add to new dataframe
        self.df[self.s1] = self.df1['Close']
        self.df[self.s2] = self.df2['Close']
        
        # find beta values for pair of stocks. This beta is not the CAPM beta, 
        # its the general beta representing the partial slope coeffiecient in a multivariate
        # (in this case univariate) regression. Given this, it also represents the min variance hedge ratio. 
        # this is a rolling regression.
        ############# need to experiment with different values for lookback window

        rolling_model = RollingOLS(self.df[self.s1], self.df[self.s2], window=self.beta_lookback)
        fitted = rolling_model.fit()

        self.df['Hedge_Ratio'] = fitted.params
        
        # the spread. For each stock_1 purchased we sell n * stock_2 where n is our hedge ratio
        # If the stocks are cointegrated, it implies the spread equation is stationary, I.E. mean and var are same over time
        # if we choose a hedge ratio such that the spread = 0, if there is cointegration the expected value of
        # the spread will stay = 0. Therefore, any deviation from this will present an opportunity of Stat Arb
        # We check for cointegration in the next method.
        self.df['Spread'] = np.log(self.df[self.s1]) - (self.df['Hedge_Ratio']*np.log(self.df[self.s2]))
        self.df['Spread_2'] = np.log(self.df[self.s2]) - (self.df['Hedge_Ratio']*np.log(self.df[self.s1]))

        return self.df
    
    def generate_cointergration(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
		''' 
        ####### can imporve this method with matrix vectorization 
        # find coint
        # tells us on a given confidence level weather the par is cointegrated and thus stationary
        adf = adfuller(self.df['Spread'].dropna())
        # print appropriate response
        if adf[0] < adf[4]['1%']:
            print('Spread is Cointegrated at 1% Significance Level: ', adf[0])
        elif adf[0] < adf[4]['5%']:
            print('Spread is Cointegrated at 5% Significance Level: ', adf[0])
        elif adf[0] < adf[4]['10%']:
            print('Spread is Cointegrated at 10% Significance Level: ', adf[0])
        else:
            print('Spread is not Cointegrated', adf[0])
        return
    
    def generate_signal(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			signas: df 
                dataframe with 1 or 0 values for long, short, and exit signals, and in position markers
		''' 
        
        #self.signals['ADF_TEST'] = 0
        self.results = []
        for i in range(self.beta_lookback, len(self.df)):
            j = i - self.beta_lookback
            test = self.df.iloc[j:i]['Spread']
            if j >= self.beta_lookback:
                adf = adfuller(test)
                if adf[0] < adf[4]['1%']:
                    #print('Spread is Cointegrated at 1% Significance Level: ', adf[0])
                    self.results.append(1)
                elif adf[0] < adf[4]['5%']:
                    #print('Spread is Cointegrated at 5% Significance Level: ', adf[0])
                    self.results.append(2)
                elif adf[0] < adf[4]['10%']:
                    #print('Spread is Cointegrated at 10% Significance Level: ', adf[0])
                    self.results.append(3)
                else:
                    #print('Spread is not Cointegrated', adf[0])
                    self.results.append(-1)
            else:
                self.results.append(0)
                self.results.append(0)
                
        # use z scores to generate buy, sell, exit signals
        # floor and ceiling threshold should be between 1.5 and 2 sigma (change depending on backtest results)
        # LONG SIGNAL = LONG THE SPREAD: BUY STOCK 1, SELL STOCK 2
        # SHORT SIGNAL = SHORT THE SPREAD: SELL STOCK 1, BUY STOCK 2
                
        # with an assumed distribution of spread ~N(0, 1), its is easy to form threshold levels 
        # these thresholds will act as signal levels 
        # Z = (X - mean) / SD
        # given time series mean and SD will be rolling, using a moving average window
        # create stock z score of the pair spread
        self.signals['Z_Score'] = ((self.df['Spread'] - self.df['Spread'].rolling(window = self.ma).mean())/
                                   (self.df['Spread'].rolling(window = self.ma).std()))

        # create prior stock z score        
        self.signals['Prior_Z_Score'] = self.signals['Z_Score'].shift(1)
        
        self.Short_Signal = False
        self.Long_Signal = False
        self.In_Short = False
        self.In_Long = False
        self.Stopped_Short = False
        self.Stopped_Long = False

        self.signals['Short_Signal'] = 0.0
        self.signals['Long_Signal'] = 0.0
        self.signals['In_Short'] = 0.0
        self.signals['In_Long'] = 0.0
        self.signals['Cover_Short'] = 0.0
        self.signals['Close_Long'] = 0.0
        self.signals['ADF'] = 0.0

        for i, j in enumerate(self.signals.iterrows()):
            current_z = j[1]['Z_Score']
            adf = self.results[i]
            self.signals.iloc[i]['ADF'] = adf
            # are we already in a short trade?
            if self.In_Short == True:
                # heave we been stopped out already?
                # define stop loss criteria
                if current_z >= self.stop_loss_short:
                    # exit trade if stop loss hit
                    # indicate we have been stopped out
                    self.In_Short = False
                    self.Stopped_Short = True
                    self.signals.iloc[i]['In_Short'] = 0.0
                    self.Short_Signal = False
                    self.signals.iloc[i]['Cover_Short'] = 1.0
                # if not stopped, have we hit close criteria?
                elif current_z <= self.exit_zscore:
                    self.In_Short = False
                    self.signals.iloc[i]['In_Short'] = 0.0
                    self.Short_Signal = False
                    self.signals.iloc[i]['Cover_Short'] = 1.0
                # if not stopped and not closed, still in trade
                else:
                    self.signals.iloc[i]['In_Short'] = 1.0
            else:
                # why are we not in a short
                # have we been stopped out or did we close position?
                if self.Stopped_Short == True:
                    self.In_Short = False
                    self.Short_Signal = False
                    self.signals.iloc[i]['In_Short'] = 0.0
                    # reset cover 
                    self.signals.iloc[i]['Cover_Short'] = 0.0
                    # if stopped, wait untill we reach exit critera to re-enter teade
                    if current_z <= self.exit_zscore:
                        self.Stopped_Short = False
                # define trade entry criteria
                elif current_z >= self.ceiling:
                    if self.hist_adf == True:
                        if adf > -1:
                            self.In_Short = True
                            self.Stopped_Short = False
                            self.signals.iloc[i]['In_Short'] = 1.0
                            self.signals.iloc[i]['Cover_Short'] = 0.0
                            if self.Short_Signal == False:
                                self.signals.iloc[i]['Short_Signal'] = 1.0
                                self.Short_Signal = True
                    else:
                        self.In_Short = True
                        self.Stopped_Short = False
                        self.signals.iloc[i]['In_Short'] = 1.0
                        self.signals.iloc[i]['Cover_Short'] = 0.0
                        if self.Short_Signal == False:
                            self.signals.iloc[i]['Short_Signal'] = 1.0
                            self.Short_Signal = True

            # are we already in a long trade?
            if self.In_Long == True:
                # define stop loss criteria
                if current_z <= self.stop_loss_long:
                    # exit trade if stop loss hit
                    # indicate we have been stopped out
                    self.In_Long = False
                    self.Stopped_Long = True
                    self.signals.iloc[i]['In_Long'] = 0.0
                    self.Long_Signal = False
                    self.signals.iloc[i]['Close_Long'] = 1.0
                # if not stopped, have we hit close criteria?
                elif current_z >= self.exit_zscore:
                    self.In_Long = False
                    self.signals.iloc[i]['In_Long'] = 0.0
                    self.Long_Signal = False
                    self.signals.iloc[i]['Close_Long'] = 1.0
                # if not stopped and not closed, still in trade
                else:
                    self.signals.iloc[i]['In_Long'] = 1.0
            else:
                # why are we not in a long
                # have we been stopped out or did we close position?
                if self.Stopped_Long == True:
                    self.In_Long = False
                    self.Long_Signal = False
                    self.signals.iloc[i]['In_Long'] = 0.0
                    # reset close
                    self.signals.iloc[i]['Close_Long'] = 0.0
                    # if stopped, wait untill we reach exit critera to re-enter teade
                    if current_z >= self.exit_zscore:
                        self.Stopped_Long = False
                # define trade entry criteria
                elif current_z <= self.floor:
                    if self.hist_adf == True:
                        if adf > -1:
                            self.In_Long = True
                            self.Stopped_Long = False
                            self.signals.iloc[i]['In_Long'] = 1.0
                            self.signals.iloc[i]['Close_Long'] = 0.0
                            if self.Long_Signal == False:
                                self.signals.iloc[i]['Long_Signal'] = 1.0
                                self.Long_Signal = True
                    else:
                        self.In_Long = True
                        self.Stopped_Long = False
                        self.signals.iloc[i]['In_Long'] = 1.0
                        self.signals.iloc[i]['Close_Long'] = 0.0
                        if self.Long_Signal == False:
                            self.signals.iloc[i]['Long_Signal'] = 1.0
                            self.Long_Signal = True
        
        self.df['Floor'] = self.floor
        self.df['Ceiling'] = self.ceiling
        self.df['Long_Stop_Loss'] = self.stop_loss_long
        self.df['Short_Stop_Loss'] = self.stop_loss_short
        self.signals['exit_zscore'] = self.exit_zscore
        
        if self.show_plot == False:
            print("show_plot = False")
        else:
            
            self.chart = ChartBuilder(self.s1, self.s2, self.start, self.end)
            self.chart.plot(self.signals, pd.DataFrame(), 'Z_Score')

        return self.signals
     