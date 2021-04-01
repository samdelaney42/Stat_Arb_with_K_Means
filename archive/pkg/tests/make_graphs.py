class ChartBuilder(object):
    
    def __init__(self, s1, s2, start, end, exit=0, long_stop = -3, short_stop = 3, long_enter = -2, short_enter = 2):
        
        self.s1 = s1
        self.s2 = s2
        self.start = start
        self.end = end
        self.exit = exit
        self.long_stop = long_stop
        self.short_stop = short_stop
        self.long_enter = long_enter
        self.short_enter = short_enter
        
        self.pair = str("{} and {} ".format(self.s1, self.s2))
        self.scale = str(" ({} - {})".format(self.start, self.end))
        
# Customized Area chart

    def plot(self, signal_data, metric_data, metric, add_signal_levels = True, add_trades = True):
        
        
        self.signal_data = signal_data
        self.metric_data = metric_data
        self.data = pd.concat([self.signal_data, self.metric_data], axis = 1)
        self.metric = str(metric) 
        self.add_signal_levels = add_signal_levels
        self.add_trades = add_trades
        self.title = self.pair + self.metric + self.scale
        self.y0 = self.data[self.metric].max()
        self.y1 = self.data[self.metric].min()
        c_line = px.line(x = self.data.index, y = self.data[self.metric], title = self.title)

        c_line.update_xaxes(
            title_text = 'Date',
            rangeslider_visible = False,
            rangeselector = dict(
                buttons = list([
                    dict(count = 1, label = '1M', step = 'month', stepmode = 'backward'),
                    dict(count = 6, label = '6M', step = 'month', stepmode = 'backward'),
                    dict(count = 1, label = 'YTD', step = 'year', stepmode = 'todate'),
                    dict(count = 1, label = '1Y', step = 'year', stepmode = 'backward'),
                    dict(step = 'all')])))
        
        if self.add_signal_levels == True:
            if self.metric == 'Z_Score':
                c_line.add_shape(type = "line", x0 = self.start, x1 = self.end, y0 = self.exit, y1 = self.exit, line=dict(color = 'orange', width = 2, dash = 'dash'))
                c_line.add_shape(type = "line", x0 = self.start, x1 = self.end, y0 = self.short_stop, y1 = self.short_stop, line=dict(color = 'red', width = 2, dash = 'dash'))
                c_line.add_shape(type = "line", x0 = self.start, x1 = self.end, y0 = self.long_stop, y1 = self.long_stop, line=dict(color = 'red', width = 2, dash = 'dash'))
                c_line.add_shape(type = "line", x0 = self.start, x1 = self.end, y0 = self.long_enter, y1 = self.long_enter, line=dict(color = 'green', width = 2, dash = 'dash'))
                c_line.add_shape(type = "line", x0 = self.start, x1 = self.end, y0 = self.short_enter, y1 = self.short_enter, line=dict(color = 'green', width = 2, dash = 'dash'))
        
        if self.add_trades == True:
            enter = self.data['Short_Signal'].gt(0.5)
            exit = self.data['Cover_Short'].gt(0.5)
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

            for index, row in ts.iterrows():
                c_line.add_shape(type = 'rect', 
                              x0 = row['first'], 
                              y0 = self.y0, 
                              x1 = row['last'], 
                              y1 = self.y1, 
                              line=dict(color="rgba(0,0,0,0)", width=3,), fillcolor='red', opacity = 0.2, layer='below')
            
            enter = self.data['Long_Signal'].gt(0.5)
            exit = self.data['Close_Long'].gt(0.5)
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
                    
            ts = pd.DataFrame(columns = ['first', 'last'])
            ts['first'] = ff
            ts['last'] = ll

            for index, row in ts.iterrows():
                c_line.add_shape(type = 'rect', 
                              x0 = row['first'], 
                              y0 = self.y0, 
                              x1 = row['last'], 
                              y1 = self.y1, 
                              line=dict(color="rgba(0,0,0,0)", width=3,), fillcolor='green', opacity = 0.2, layer='below')

        c_line.update_yaxes(title_text = self.metric)
        c_line.update_layout(showlegend = False,
            title = {
                'text': self.title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        c_line.show()
       