class MakePortfolio(object):
	def __init__(self):
		
		self.yeet = 1
		
		
		
	def generate_trades(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			trades: df
                dataframe with negative or positive price values for each stock depending on long or short trade
                total indicates net value of trade
		''' 
        self.trades['Positions'] = self.signals['In_Long'] - self.signals['In_Short']
        # Long stock shows negative value = price to represent cash outflow of bought share
        # short stock shows positive value = price to represent cash inflow of borrowed shares
        self.trades[self.s1] = -1 * (self.df[self.s1] * self.trades['Positions'])
        self.trades[self.s2] = (self.df[self.s2] * self.trades['Positions'])
        # Total shows current cumulative value of positions
        self.trades['Total'] = self.trades[self.s1] + self.trades[self.s2]

        return self.trades
    
    def generate_portfolio(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			portfolio: df
                dataframe with return values for trades, and overall portfolio vlaue
		''' 
        
        # create percentage returns stream
        # find daily change of total market value of positions
        self.portfolio['Returns'] = self.trades['Total'].pct_change()
        self.portfolio['Returns'].fillna(0.0, inplace = True)
        # account for % changes where start = 0 as this would be infinite
        self.portfolio['Returns'].replace([np.inf, -np.inf], 0.0, inplace = True)
        self.portfolio['Returns'].replace(-1.0, 0.0, inplace = True)
        
        # getting equity curve
        self.portfolio['Cumulative_Returns'] = (self.portfolio['Returns'] + 1.0).cumprod()
        self.portfolio['Portfolio_Value'] = (self.allocation * self.portfolio['Cumulative_Returns'])
        self.portfolio['Portfolio_Returns'] = self.portfolio['Portfolio_Value'].pct_change()
        self.portfolio['Allocation'] = self.allocation
        
        self.max_returns = np.fmax.accumulate(self.portfolio['Returns'])
        self.portfolio['Max_Drawdown'] = (self.portfolio['Returns'] / self.max_returns) - 1
        
        self.portfolio['In_Long'] = self.signals['In_Long']
        self.portfolio['In_Short'] = self.signals['In_Short']
        
        # plot portfolio valuation
        if self.show_plot == False:
            print("show_plot = False")
        else:
            
            self.chart = ChartBuilder(self.s1, self.s2, self.start, self.end)
            self.chart.plot(self.signals, self.portfolio, 'Portfolio_Value')
            
        return self.portfolio
    
    def generate_order_book(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			book: df
                dataframe showing dates that a buy or sell order is made and for which stock at which quantity
		''' 
        
        self.book = pd.DataFrame(index = range(len(self.signals)))
        #self.book = self.bk
        self.book['Date'] = ""
        self.book['Position_Type'] = ""
        self.book['Stock'] = ""
        self.book['Order_Type'] = ""
        self.book['Qty'] = 0
        self.book['Price'] = 0
        self.book['Profit/Loss'] = 0
        count = 0
        for i, j in enumerate(self.signals.iterrows()):
            
            if self.signals.iloc[i]['Long_Signal'] == 1.0:
                # set type of order
                self.book.loc[count, 'Date'] = self.signals.index[i]
                self.book.loc[count, 'Position_Type'] = "Long"
                self.book.loc[count, 'Stock'] = self.s1
                self.book.loc[count, 'Order_Type'] = "Buy"
                self.long_quant = np.floor((self.allocation) / self.df.iloc[i][self.s1])
                self.book.loc[count, 'Qty'] = self.long_quant
                self.enter_long_stock_1_price = self.df.iloc[i][self.s1]
                self.book.loc[count, 'Price'] = self.enter_long_stock_1_price
                
                self.book.loc[count+1, 'Date'] = self.signals.index[i]
                self.book.loc[count+1, 'Position_Type'] = "Long"
                self.book.loc[count+1, 'Stock'] = self.s2
                self.book.loc[count+1, 'Order_Type'] = "Sell"
                self.short_quant = np.floor(self.long_quant * self.df.iloc[i]['Hedge_Ratio'])
                self.book.loc[count+1, 'Qty'] = self.short_quant
                self.enter_long_stock_2_price = self.df.iloc[i][self.s2]
                self.book.loc[count+1, 'Price'] = self.enter_long_stock_2_price
                count += 2
            elif self.signals.iloc[i]['Close_Long'] == 1.0:
                self.book.loc[count, 'Date'] = self.signals.index[i]
                self.book.loc[count, 'Position_Type'] = "Close Long"
                self.book.loc[count, 'Stock'] = self.s1
                self.book.loc[count, 'Order_Type'] = 'Sell'
                self.book.loc[count, 'Qty'] = self.long_quant
                self.exit_long_stock_1_price = self.df.iloc[i][self.s1]
                self.book.loc[count, 'Price'] = self.exit_long_stock_1_price
                self.book.loc[count, 'Profit/Loss'] = ((self.exit_long_stock_1_price - self.enter_long_stock_1_price)
                                                       *self.long_quant)
                
                self.book.loc[count+1, 'Date'] = self.signals.index[i]
                self.book.loc[count+1, 'Position_Type'] = "Close Long"
                self.book.loc[count+1, 'Stock'] = self.s2
                self.book.loc[count+1, 'Order_Type'] = 'Buy'
                self.book.loc[count+1, 'Qty'] = self.short_quant
                self.exit_long_stock_2_price = self.df.iloc[i][self.s2]
                self.book.loc[count+1, 'Price'] = self.exit_long_stock_2_price
                self.book.loc[count+1, 'Profit/Loss'] = -1*((self.exit_long_stock_2_price - self.enter_long_stock_2_price)
                                                            *self.short_quant)
                count += 2
            
            if self.signals.iloc[i]['Short_Signal'] == 1.0:
                # set type of order
                self.book.loc[count, 'Date'] = self.signals.index[i]
                self.book.loc[count, 'Position_Type'] = "Short"
                self.book.loc[count, 'Stock'] = self.s2
                self.book.loc[count, 'Order_Type'] = "Buy"
                self.long_quant = np.floor((self.allocation) / self.df.iloc[i][self.s2])
                self.book.loc[count, 'Qty'] = self.long_quant
                self.enter_long_stock_2_price = self.df.iloc[i][self.s2]
                self.book.loc[count, 'Price'] = self.enter_long_stock_2_price
                
                self.book.loc[count+1, 'Date'] = self.signals.index[i]
                self.book.loc[count+1, 'Position_Type'] = "Short"
                self.book.loc[count+1, 'Stock'] = self.s1
                self.book.loc[count+1, 'Order_Type'] = "Sell"
                self.short_quant = np.floor(self.long_quant * self.df.iloc[i]['Hedge_Ratio'])
                self.book.loc[count+1, 'Qty'] = self.short_quant
                self.enter_long_stock_1_price = self.df.iloc[i][self.s1]
                self.book.loc[count+1, 'Price'] = self.enter_long_stock_1_price
                count += 2
                
            elif self.signals.iloc[i]['Cover_Short'] == 1.0:
                self.book.loc[count, 'Date'] = self.signals.index[i]
                self.book.loc[count, 'Position_Type'] = "Close Short"
                self.book.loc[count, 'Stock'] = self.s2
                self.book.loc[count, 'Order_Type'] = 'Sell'
                self.book.loc[count, 'Qty'] = self.long_quant
                self.exit_long_stock_2_price = self.df.iloc[i][self.s2]
                self.book.loc[count, 'Price'] = self.exit_long_stock_2_price
                self.book.loc[count, 'Profit/Loss'] = ((self.exit_long_stock_2_price - self.enter_long_stock_2_price)
                                                       *self.long_quant)
                
                self.book.loc[count+1, 'Date'] = self.signals.index[i]
                self.book.loc[count+1, 'Position_Type'] = "Close Short"
                self.book.loc[count+1, 'Stock'] = self.s1
                self.book.loc[count+1, 'Order_Type'] = 'Buy'
                self.book.loc[count+1, 'Qty'] = self.short_quant
                self.exit_long_stock_1_price = self.df.iloc[i][self.s1]
                self.book.loc[count+1, 'Price'] = self.exit_long_stock_1_price
                self.book.loc[count+1, 'Profit/Loss'] = -1*((self.exit_long_stock_1_price - self.enter_long_stock_1_price)
                                                            *self.short_quant)
                count += 2
        
        self.book['Cum_Sum_P_L'] = self.book['Profit/Loss'].cumsum()
        self.book = self.book.drop(self.book.index[count:])
        return self.book
        
    def generate_metrics(self):
        '''
		parameters
		----------
			self: object 
                statarb object
		returns
		----------
			metrics: df
                dataframe of performance metrics
		''' 
        # calculate summary statistics
        self.mu = (self.portfolio['Returns'].mean())
        self.sigma = (self.portfolio['Returns'].std())
        self.sharpe = (self.mu - 0.005) / self.sigma
        # where True, yield x, otherwise yield y
        self.wins = (np.where(self.portfolio['Cumulative_Returns'] > 0.0, 1.0, 0.0)).sum()
        self.losses = (np.where(self.portfolio['Cumulative_Returns'] < 0.0, 1.0, 0.0)).sum()
        if self.losses == 0:
            self.total_trades = self.wins
        else:
            self.total_trades = self.wins + self.losses
        # win loss ratio
        if self.losses == 0:
            self.wl_ratio = 1
        else:
            self.wl_ratio = (self.wins / self.losses)
        # probability of win and loss
        self.p_win = (self.wins / self.total_trades)
        if self.losses == 0:
            self.p_loss = 0
        else:
            self.p_loss = (self.losses / self.total_trades)
        # avg win / loss return
        self.avg_win_return = (self.portfolio['Cumulative_Returns'] > 0.0).mean()
        self.avg_loss_return = (self.portfolio['Cumulative_Returns'] < 0.0).mean()
        # payout ratio
        if self.avg_loss_return == 0:
            self.payout_ratio = 1
        else:
            self.payout_ratio = (self.avg_win_return / self.avg_loss_return)
        
        self.difference_in_years = relativedelta(self.end, self.start).years

        self.metrics['CAGR'] = (((self.portfolio.iloc[-1]['Portfolio_Value']/self.portfolio.iloc[0]['Portfolio_Value'])
                                 **(1/self.difference_in_years)) - 1)
        self.metrics['Sharpe Ratio'] = self.sharpe
        self.metrics['Wins'] = self.wins
        self.metrics['P(Wins)'] = self.p_win
        self.metrics['Avg_Win_Return'] = self.avg_win_return
        self.metrics['Losses'] = self.losses
        self.metrics['P(Loss)'] = self.p_loss
        self.metrics['Avg_Loss_Return'] = self.avg_loss_return
        self.metrics['WL_Ratio'] = self.wl_ratio
        
        return self.metrics
    
    def trade_summary(self):
        enter = self.signals['Short_Signal'].gt(0.5)
        exit = self.signals['Cover_Short'].gt(0.5)
        shorts = pd.concat([enter, exit], axis=1)
        shorts = shorts[~((~shorts).all(axis=1))]
        shorts.reset_index(level=0, inplace=True)
        ff = []
        ll = []
        f = shorts.loc[shorts['Short_Signal'] == True, 'Date']
        l = shorts.loc[shorts['Cover_Short'] == True, 'Date']
        run_f = True
        run_l = True
        if shorts.iloc[0]['Short_Signal'] == False:
            ff.append(self.start)
            for i in f:
                ff.append(i)
            run_f = False
        elif shorts.iloc[-1]['Short_Signal'] == True:
            for i in l:
                ll.append(i)
            ll.append(self.end)
            run_l = False

        if run_f == True:
            for i in f:
                ff.append(i)
        if run_l == True:
            for i in l:
                ll.append(i)

        ts = pd.DataFrame(columns = ['first', 'last'])
        ts['first'] = ff
        ts['last'] = ll
        ts['type'] = 'short'
        
        enter = self.signals['Long_Signal'].gt(0.5)
        exit = self.signals['Close_Long'].gt(0.5)
        longs = pd.concat([enter, exit], axis=1)
        longs = longs[~((~longs).all(axis=1))]
        longs.reset_index(level=0, inplace=True)
        ff = []
        ll = []
        f = longs.loc[longs['Long_Signal'] == True, 'Date']
        l = longs.loc[longs['Close_Long'] == True, 'Date']
        run_f = True
        run_l = True
        if longs.iloc[0]['Long_Signal'] == False:
            ff.append(self.start)
            for i in f:
                ff.append(i)
            run_f = False
        elif longs.iloc[-1]['Long_Signal'] == True:
            for i in l:
                ll.append(i)
            ll.append(self.end)
            run_l = False

        if run_f == True:
            for i in f:
                ff.append(i)
        if run_l == True:
            for i in l:
                ll.append(i)

        tl = pd.DataFrame(columns = ['first', 'last'])
        tl['first'] = ff
        tl['last'] = ll
        tl['type'] = 'long'
        
        ts = ts.append(tl, ignore_index = True)
        
        return ts