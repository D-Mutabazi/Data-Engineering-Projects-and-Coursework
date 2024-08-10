import plotly.graph_objects as go   #plot candle stick data - plotly library
import pandas as pd                 #pandas are used to analyze data

#draw candle stick data
def candleChart(df):
    fig = go.Figure(data=[go.Candlestick(x=df['Time'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

df = pd.read_csv('./Data/EURUSD_H1.csv') #check for if file is comma/tab/space ..etc separated

print(df.head())
df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')  #if time included in date
# df['Time'] = pd.to_datetime(df['Time']) # if time not included
print(df.head())

#candlestick chart
candleChart(df[-1000:])