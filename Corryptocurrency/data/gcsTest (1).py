# -*- coding: utf-8 -*-
"""
Created on Wed May 10 02:04:50 2023

@author: kimky
"""
import requests
import pandas as pd
from pytz import timezone
import time
import random
from datetime import datetime, timedelta
import pytz
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense


tz = timezone('Asia/Seoul')

def get_top10_coins():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': '10',
        'page': '1',
        'sparkline': 'false'
    }
    response = requests.get(url, params=params)
    coins = response.json()
    if coins:
        coin_ids = [coin['id'] for coin in coins if 'id' in coin]
        return coin_ids
    else:
        return []

def get_coin_prices(coin_id):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '7'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'prices' in data:
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(tz)
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['coin_id'] = coin_id
        return df
    else:
        return pd.DataFrame()

def get_top10_prices():
    coin_ids = get_top10_coins()
    dfs = []
    for coin_id in coin_ids:
        while len(dfs) < 10:
            df = get_coin_prices(coin_id)
            if not df.empty:
                dfs.append(df)
                break
            else:
                time.sleep(0.1)
    return pd.concat(dfs)


# Get the latest top 10 prices
data = get_top10_prices()
data = data.to_json(date_format='iso', orient='records')
data = pd.read_json(data)

data_dict = {}
for coin_id in data['coin_id'].unique():
    coin_data = data.loc[data['coin_id'] == coin_id]
    data_dict[coin_id] = coin_data

coin_names = list(data['coin_id'].unique())


random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

models = {}

for coin_id, coin_data in data_dict.items():
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(coin_data['price']).reshape(-1, 1))

    # Create the training dataset
    train_data = scaled_data[0:int(len(coin_data) * 0.8), :]

    # Split the data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)
    # Store the trained model
    save_model = {'model': model, 'scaler': scaler}
    # Store the best model for this coin
    models[coin_id] = save_model

def predict_price(coin_id):
    # Get the model and scaler for the requested coin
    model_data = models.get(coin_id, None)
    if not model_data:
        return None

    # Get the data for the requested coin
    coin_data = data_dict[coin_id]

    # Get the scaler for this coin's data
    scaler = model_data['scaler']

    # Prepare the test data
    test_data = scaler.transform(np.array(coin_data['price']).reshape(-1, 1))[int(len(coin_data)*0.8)-60:, :]

    # Split the data into x_test and y_test datasets
    x_test = []

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the predicted price
    model = model_data['model']
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Get the next hour predicted price
    next_hour_price = predicted_price[-1][0]

    print(coin_id + ": " + str(next_hour_price))


data0 = data_dict[coin_names[0]]
data1 = data_dict[coin_names[1]]
data2 = data_dict[coin_names[2]]
data3 = data_dict[coin_names[3]]
data4 = data_dict[coin_names[4]]
data5 = data_dict[coin_names[5]]
data6 = data_dict[coin_names[6]]
data7 = data_dict[coin_names[7]]
data8 = data_dict[coin_names[8]]
data9 = data_dict[coin_names[9]]

data0 = data0.to_json(date_format='iso', orient='records')
data1 = data1.to_json(date_format='iso', orient='records')
data2 = data2.to_json(date_format='iso', orient='records')
data3 = data3.to_json(date_format='iso', orient='records')
data4 = data4.to_json(date_format='iso', orient='records')
data5 = data5.to_json(date_format='iso', orient='records')
data6 = data6.to_json(date_format='iso', orient='records')
data7 = data7.to_json(date_format='iso', orient='records')
data8 = data8.to_json(date_format='iso', orient='records')
data9 = data9.to_json(date_format='iso', orient='records')

coin0_predict = predict_price(coin_names[0])
coin1_predict = predict_price(coin_names[1])
coin2_predict = predict_price(coin_names[2])
coin3_predict = predict_price(coin_names[3])
coin4_predict = predict_price(coin_names[4])
coin5_predict = predict_price(coin_names[5])
coin6_predict = predict_price(coin_names[6])
coin7_predict = predict_price(coin_names[7])
coin8_predict = predict_price(coin_names[8])
coin9_predict = predict_price(coin_names[9])