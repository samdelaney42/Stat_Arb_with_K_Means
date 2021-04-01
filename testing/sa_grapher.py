import plotly.express as px
import plotly.graph_objects as go


class Grapher(object):

	def __init__(self, data):

		self.data = data

	def show_z_score(self):

		c_line = px.line(self.data, x = self.data.index, y = [self.data['Z_Score'],
																self.data['Mean'],
																self.data["stop_loss_short"],
																self.data["enter_short"],
																self.data["stop_loss_long"],
																self.data["enter_long"]], title = 'stocks')

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

		c_line.update_yaxes(title_text = 'stocks')
		c_line.update_layout(showlegend = True,
		    title = {
		        'text': 'stocks',
		        'y':0.9,
		        'x':0.5,
		        'xanchor': 'center',
		        'yanchor': 'top'})

		c_line.show()


if __name__ == '__main__':

	main()