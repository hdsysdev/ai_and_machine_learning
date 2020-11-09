import numpy
import pandas
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout, Dense
from sklearn.metrics import mean_absolute_error

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()
# Reindexing after removing NaN rows
df.reset_index(drop=True, inplace=True)

df.loc[:, "Date"] = df.Timestamp.values.astype(dtype='datetime64[s]')
data = pandas.DataFrame(columns=["Date", "Close"], index=range(0, len(df)))

for i in range(0, len(data)):
    data["Date"][i] = df["Date"][i]
    data["Close"][i] = df["Close"][i]

data.set_index('Date', inplace=True, drop=True)
print(data.head())
# Drop rows before 01/01/2019
data_train = data[(data["Date"] <= pandas.Timestamp("01/01/2020").timestamp())]
data_test = data[(data["Date"] >= pandas.Timestamp("01/01/2020").timestamp())]

plot.plot(data_train['Close'], label="Train Data")
plot.plot(data_test['Close'], label="Test Data")
plot.ylabel("Bitcoin Price in $")
plot.legend(loc="best")
plot.show()
# Create new column with python datetime to plt graph

# plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
# plot.title("Bitcoin closing price from the start of 2017")
# plot.ylabel("Closing Price in $")
# plot.xlabel("Date")
# plot.xticks(rotation=40)
# plot.grid(True)
# plot.show()
#
# test_x = data_test[["Timestamp"]]
# test_y = data_test[["Close"]]
#
# x = pandas.DataFrame(data_train["Timestamp"])
# y = pandas.DataFrame(data_train["Close"])
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)
# minMaxScaler = MinMaxScaler()
# Not using scaler since you wouldn't know the min and max values in live environment
# scaled_x_train = minMaxScaler.fit_transform(x_train)
# scaled_x_test = minMaxScaler.fit_transform(x_test)
# scaled_y_train = minMaxScaler.fit_transform(y_train)
# scaled_y_test = minMaxScaler.fit_transform(y_test)
#
# scaled_test_x = minMaxScaler.fit_transform(test_x)
# scaled_test_y = minMaxScaler.fit_transform(test_y)




# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
# plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], scaled_y_test[::50], s=1, label="Train")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
# lstm = Sequential()
# lstm.add(LSTM(units=100, input_shape=x_train.shape))
# lstm.add(Dropout(0.2))
# lstm.add(Dense(1))
# # lstm.add(Activation('linear'))
# lstm.compile(loss="mse", optimizer="adam")
# lstm.summary()
#
# history = lstm.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, shuffle=True)
# plot.plot(history.history['loss'], label='train')
# plot.legend()
# plot.show()
#
# targets = data_test["Close"][5:]
# preds = lstm.predict(x_test).squeeze()
# print(mean_absolute_error(preds, y_test))

# Using scatter plot as plot_date function's linewidth property isn't working
# plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
#              scaled_test_y, label="Actual",
#              s=12)
# plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
#              y_predicted, label="Degree 8" + " R^2 Score: " + str(format(score, ".3f")),
#              s=12)
# print("Polynomial Degree 8" + " Score: " + str(score))

plot.legend(loc="lower right", fontsize="small")
plot.show()
