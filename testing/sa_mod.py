import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
import yahoo_finance as y_f
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from datetime import date
from statsmodels.tsa.api import adfuller
import matplotlib.pyplot as plt
from pypfopt.discrete_allocation import DiscreteAllocation

class StatArb(object):
    
    def __init__(self, stock_1, stock_2, start, end, long_ma, short_ma, allocation):
        
        self.stock_1 = stock_1
        self.stock_2 = stock_2
        self.start = start
        self.end = end
        self.long_ma = long_ma
        self.short_ma = short_ma
        self.allocation = allocation
        self._valid_pair = True
        
        self.stock_1_prices = yf.download(self.stock_1, self.start, self.end)['Adj Close']
        if len(self.stock_1_prices) == 0:
            self._valid_pair = False

        self.stock_2_prices = yf.download(self.stock_2, self.start, self.end)['Adj Close']
        if len(self.stock_2_prices) == 0:
            self._valid_pair = False
            

        self.index = self.stock_1_prices.index
        
        self.data = pd.DataFrame(index = self.index, columns = [self.stock_1, self.stock_2])
        self.data[self.stock_1] = self.stock_1_prices
        self.data[self.stock_2] = self.stock_2_prices

    def create(self):

        if self._valid_pair:
            self.prepared = self.prepare_data()
            self.tested = self.adf_test()
            self.sigs = self.generate_signals()
            self.summary = self.trade_summary()
            self.returns = self.generate_returns()
        else:
            print("Couldn't get data for {}/{}".format(self.stock_1, self.stock_2))
            return False
        
        
    def prepare_data(self):

        X1 = np.log(self.stock_1_prices)
        X2 = np.log(self.stock_2_prices)
        x1 = sm.add_constant(X1)
        rolling_model = RollingOLS(X2, X1, window=self.long_ma)
        fitted_model = rolling_model.fit()
        
        self.data["Hedge_Ratio"] = fitted_model.params
        
        self.data["Spread"] = np.log(self.data[self.stock_2]) - (self.data["Hedge_Ratio"]*np.log(self.data[self.stock_1]))
        
        # Z = (X - M)/std
        self.data["X"] = self.data["Spread"]
        self.data["Mean"] = self.data["Spread"].rolling(window=self.long_ma).mean()
        self.data["StDev"] = self.data["Spread"].rolling(window=self.long_ma).std()
        
        self.data["Z_Score"] = (self.data["X"] - self.data["Mean"]) / self.data["StDev"] 
        
        self.data["stop_loss_short"] = 3
        self.data["enter_short"] = 2
        self.data["stop_loss_long"] = -3
        self.data["enter_long"] = -2
        self.data["Rolling_ADF"] = 0 

        return self.data
    
    def adf_test(self):
        
        adf_results = []
        for i in range(self.long_ma, len(self.data)):
            j = i - self.long_ma
            if j > self.long_ma:
                test_period = self.data.iloc[j:i]["Spread"]
                adf = adfuller(test_period)
                if adf[0] < adf[4]['1%']:
                    adf_results.append(1)
                elif adf[0] < adf[4]['5%']:
                    adf_results.append(2)
                elif adf[0] < adf[4]['10%']:
                    adf_results.append(3)
                else:
                    adf_results.append(-1)
            else:
                adf_results.append(0)
                
        self.data_c = self.data[self.long_ma:].copy()
        self.data_c["Rolling_ADF"] = adf_results
        self.data_c = self.data_c.dropna()
        
        return self.data_c
    
    def generate_signals(self):
        p = self.stock_1 + "/" + self.stock_2
        self.l = "In_Long_{}".format(p)
        self.s = "In_Short_{}".format(p)

        self.signals = pd.DataFrame(index = self.data.index)
        self.signals['Hedge_Ratio'] = self.data['Hedge_Ratio']
        self.signals[self.l] = 0
        self.signals[self.s] = 0
        
        in_short = False
        stopped_short = False
        closed_short = False
        in_long = False
        stopped_long = False
        closed_long = False

        stop_loss_short = 3
        enter_short = 2
        stop_loss_long = -3
        enter_long = -2
        
        for index, row in enumerate(self.data.iterrows()):
            
            z = row[1]["Z_Score"]
            exit = row[1]["Mean"]
            i = row[0]
            
            if in_short:
                if z > stop_loss_short:
                    stopped_short = True
                    in_short = False
                    self.signals.at[i, self.s] = 0
                elif z < exit:
                    closed_short = True
                    in_short = False
                    self.signals.at[i, self.s] = 0
                else:
                    self.signals.at[i, self.s] = 1
            else:
                if stopped_short:
                    if z > enter_short:
                        self.signals.at[i, self.s] = 0
                    else:
                        self.signals.at[i, self.s] = 0
                        stopped_short = False
                elif z > enter_short:
                    self.signals.at[i, self.s] = 1
                    in_short = True
                    
            if in_long:
                if z < stop_loss_long:
                    stopped_long = True
                    in_long = False
                    self.signals.at[i, self.l] = 0
                elif z > exit:
                    closed_Long = True
                    in_long = False
                    self.signals.at[i, self.l] = 0
                else:
                    self.signals.at[i, self.l] = 1
            else:
                if stopped_long:
                    if z < enter_long:
                        self.signals.at[i, self.l] = 0
                    else:
                        self.signals.at[i, self.l] = 0
                        stopped_long = False
                elif z < enter_long:
                    self.signals.at[i, self.l] = 1
                    in_long = True

        self.signals[self.stock_1] = self.data[self.stock_1][self.signals.index[0]:]
        self.signals[self.stock_2] = self.data[self.stock_2][self.signals.index[0]:]

        return self.signals
                    

    def trade_summary(self):
        
        def trade_type(t_type):
            t = [x for x in range(1, len(self.signals)) if (self.signals.iloc[x-1][t_type] == 0 and self.signals.iloc[x][t_type] == 1) or (self.signals.iloc[x-1][t_type] == 1 and self.signals.iloc[x][t_type] == 0) ]
            trades = self.signals.iloc[t][:]
            s1_size_title = "{}_Size".format(self.stock_1)
            s2_size_title = "{}_Size".format(self.stock_2)
            s2_size = (self.allocation // 2) // self.signals[self.stock_2]
            s1_size = round((self.signals["Hedge_Ratio"] * s2_size), 0)
            trades[s1_size_title] = s1_size
            trades[s2_size_title] = s2_size
            return trades 
            #self.signals.index[t].tolist()
        
        return trade_type(self.s), trade_type(self.l)

    def generate_returns(self):

        self.data['Position'] = self.signals[self.l] - self.signals[self.s]

        self.data[self.stock_1] = -1 * (self.data[self.stock_1] * self.data['Position'])
        self.data[self.stock_2] = (self.data[self.stock_2] * self.data['Position'])

        self.data['Total'] = self.data[self.stock_1] + self.data[self.stock_2]

        self.returns = pd.DataFrame(index = self.data.index)

        self.returns["Hedge_Ratio"] = self.data["Hedge_Ratio"]

        self.returns['Returns'] = self.data['Total'].pct_change()
        self.returns['Returns'].fillna(0.0, inplace = True)

        self.returns['Returns'].replace([np.inf, -np.inf], 0.0, inplace = True)
        self.returns['Returns'].replace(-1.0, 0.0, inplace = True)
        
        self.returns['Cumulative_Returns'] = (self.returns['Returns'] + 1.0).cumprod()
        self.returns['Trade_Value'] = (self.allocation * self.returns['Cumulative_Returns'])
        
        self.max_returns = np.fmax.accumulate(self.returns['Returns'])
        self.returns['Max_Drawdown'] = (self.returns['Returns'] / self.max_returns) - 1
        
        self.returns[self.l] = self.signals[self.l]
        self.returns[self.s] = self.signals[self.s]

        self.returns['t'] = self.data['Position']

        return self.returns

    def foward_test(self):

    	n = 10000
    	d = 20
    	T = 1.
    	times = np.linspace(0., T, n)
    	dt = times[1] - times[0]
    	# BT2 - BT1 ~ N(0, T2-T1)
    	dB = np.sqrt(dt) * np.random.normal(size=(n-1, d))
    	B0 = np.zeros(shape=(1, d))
    	B = np.concatenate((B0, np.cumsum(dB, axis = 0)), axis = 0)
    	plt.plot(times, B)
    	plt.show()

if __name__ == '__main__':

	main()



