##########Modules##########
import numpy as np #Used to perform math operations on the data
import matplotlib.pyplot as plt #Used to plot the data
import pandas as pd #Used to analyse the data
import pandas_datareader as web #Used to fetch data from data soures on the internet, I used to to fetch data from Yahoo Finance
import datetime as dt # set a date + get current date

from sklearn.preprocessing import MinMaxScaler #Used in the macine learning prediction part
from tensorflow.keras.layers import Dense, Dropout, LSTM #Used for fast numerical computing as we have a lot of prices to track
from tensorflow.keras.models import Sequential #Used for fast numerical computing as we have a lot of prices to track

crypto_currency = "BTC" #Setting the crypto to fetch from Yahoo Finance eg. BTC, ETH, XRP, BNB, USDT, DOGE, LTC, BCH, XLM
against_currency = "USD" #Setting the currency to convert crypto into from Yahoo Finance eg. EUR, USD, GBP

start = dt.datetime(2020, 1, 1) #Setting the initial data to start data collection
end = dt.datetime.now() #Setting the end date

data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", start, end) #Fetching the Crypto prices from Yahoo finance from the start to end date

##########Prepare Data##########

####This is the data that the program fetches from Yahoo Finance, if you would like to see it comment out the rest of the program and uncomment the print statement
####I am interested in the close column as that is the end of the day, so I fetch the data of each day from the last possible value available. This is done so the data is consistant and we don't measure at 11pm one day then 2am the next
#print(data.head)

scaler = MinMaxScaler(feature_range=(0, 1)) #Scale down data between 0 and 1 so the neural network has an easier time comupting it, this will speed up the machine learning 
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) #Fecthing the data from the close column as this is the end of the day

prediction_days = 60 #The number of days the program looks in the past to predict what the value is
future_day = 5 #The number of days in advance eg. 0 days - 60th day it will guess the the 65th day

x_train, y_train = [], [] #The data list so we can plot the days later

for x in range(prediction_days, len(scaled_data) - future_day): #Start at the 60th day, looking back to the 0 day value all the way to the end of the days. - future_day is used to move all the data back 5 days so the data is accurate on the plot
    x_train.append(scaled_data[x-prediction_days:x, 0]) #Chunks of 60 days
    y_train.append(scaled_data[x+future_day, 0]) #The day number of days we want to predict in the future
    
x_train, y_train = np.array(x_train), np.array(y_train) #Turning it into a numpy array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #Reshape the x_train, adds another dimension to the data

##########Create neural network##########

model = Sequential() #A basic sequential model

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))) #LSTM = Long Short Term Memory layers are used to memorise the sequential data, feeds data back into the neural network
model.add(Dropout(0.2)) #Prevents the network from being overfit with data
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Gets the Mean of the numbers, so we have 1 value instead of multiple values. This is the price prediction

model.compile(optimizer='adam', loss='mean_squared_error') #Compiling the model
model.fit(x_train, y_train, epochs=25, batch_size=64) #This is training the model with the given x_train and y_train values. epochs are the number of iterations a model runs, more epochs gives more acurate data however at a cost of speed

##########Testing the model##########

test_start = dt.datetime(2020, 1, 1) #Setting the initial data to start data collection
test_end = dt.datetime.now() #Setting the end date

test_data = web.DataReader(f"{crypto_currency}-{against_currency}", "yahoo", test_start, test_end) # Fetching the test data (the crypto prices), this data hasn't been run throught machine learning to predict the price
actual_prices = test_data['Close'].values #These are the actual prices of the crypto, getting the end of day values like above

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0) #Combine the test dataset with the predicted data set

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values #These are the values used for the predictions
model_inputs = model_inputs.reshape(-1, 1) #Reshaping the data to be plotted
model_inputs = scaler.fit_transform(model_inputs) #We need to scale the data back to 0 and 1 so it matches the data above

x_test = [] #List of predictions

for x in range(prediction_days, len(model_inputs)): #Making predictions using the trained model for the 60 day blocks
    x_test.append(model_inputs[x-prediction_days:x, 0]) #Starting at 0 because we go back 60 days all the way up to the 60th day

x_test = np.array(x_test) #Turning it into a numpy array
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #Reshaping to add a third dimension, to match the data above

prediction_prices = model.predict(x_test) #Now we fetch the predictions that the model makes
prediction_prices = scaler.inverse_transform(prediction_prices) #Now we the actual values that aren't scaled by the model

##########Plotting the data ##########
plt.plot(actual_prices, color='black', label='Actual Prices') #Plot the actual price of the crypto in black
plt.plot(prediction_prices, color='blue', label='Prediction Prices') #Plot the prediction price of the crypto in blue
plt.title(f"{crypto_currency} price prediction") #Add a title to the plot
plt.xlabel('Days') #Add Days label to the x axis
plt.ylabel(against_currency) #Add Currency to the y axis
plt.legend(loc='upper left') #Add a legend for the plot onto the top left corner
plt.show() #Plot the data with the above variables

