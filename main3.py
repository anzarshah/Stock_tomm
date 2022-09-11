import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from cProfile import label
from pickletools import optimize
from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas_datareader as web
import datetime as dt
from datetime import timedelta
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout,LSTM




START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

presentday = date.today()
tomorrow = presentday + timedelta(1)

st.title("Stock Prediction App (Tommorow's Prediction) ")


stocks = ('TITAN.NS', 'TCS.NS', 'MARUTI.NS', 'NESTLEIND.NS','TATASTEEL.NS','ITC.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)



@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data1 = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data1.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()



data_load_state1 = st.text('The model is being trained for selected_stock data')
data_load_state2 = st.text('THIS TAKE Around 30-40 SECOND')

start = dt.datetime(2012,1,1) 
end = dt.datetime(2020,1,1)

data = web.DataReader(selected_stock , 'yahoo' , start , end)


#preapare the data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_date = 60

x_train = []

y_train = []

for x in range(prediction_date,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_date:x,0])
    y_train.append(scaled_data[x,0])

x_train,y_train  = np.array(x_train) , np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#build model or importing model
#build model

model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #prediction of next date closing

model.compile(optimizer='adam',loss='mean_squared_error')


model.fit(x_train,y_train,epochs=6,batch_size=32)


#TESTING MODEL ACCURACY

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(selected_stock,'yahoo',test_start,test_end)

actual_price = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_date:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# x_test = []
# # Make prediction on  test data

# for x in range(prediction_date,len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_date:x,0])

# x_test  = np.array(x_test)
# x_test = np.reshape(x_test,(x_test.shape[0],x_test[1],1))


# prediction_prices = model.predict(x_test)

# prediction_prices = scaler.inverse_transform(prediction_prices)

# # plot the prediction
# fig = plt.figure(figsize = (10, 5))
# plt.plot(actual_price,color="black",label=f"Actual {company} price")
# plt.plot(prediction_prices,color="Green",label=f"Predicted {company} price")

# plt.title(f"{company} Share price")

# plt.xlabel('Time')
# plt.ylabel(f"{company} Share price")
# st.pyplot(fig)

# plt.legend()

# plt.show()


real_data = [model_inputs[len(model_inputs)+1-prediction_date:len(model_inputs+1),0]]

real_data = np.array(real_data)

real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))


# print(scaler.inverse_transform(real_data))

prediction = model.predict(real_data)

prediction = scaler.inverse_transform(prediction)


st.write(prediction)

data_load_state1.text("THE PREDICTION FOR")

data_load_state2.text(tomorrow.strftime('%d-%m-%Y'))






