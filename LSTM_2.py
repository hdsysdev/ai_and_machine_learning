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

df.index = df["Date"]
df.drop("Date", axis=1, inplace=True)
# Drop rows before 01/01/2019
data_train = df[(df.index <= "01/01/2020")]
data_test = df[(df.index >= "01/01/2020")]

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

# Not using scaler since you wouldn't know the min and max values in live environment
minMaxScaler = MinMaxScaler()
data_scaled = minMaxScaler.fit_transform(df)

x = pandas.DataFrame(data_train["Timestamp"])
y = pandas.DataFrame(data_train["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

x_train, y_train = numpy.array(x_train), numpy.array(y_train)
x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


x_test = numpy.array(x_test)
x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
# plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], scaled_y_test[::50], s=1, label="Train")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
lstm = Sequential()
lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm.add(LSTM(units=50))
lstm.add(Dense(1))
# lstm.add(Activation('linear'))
lstm.compile(loss="mse", optimizer="adam")
lstm.summary()
history = lstm.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
lstm.save("lstm.model")
# history = lstm.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1, shuffle=True)
plot.plot(history.history['loss'], label='train')
plot.legend()
plot.show()
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

# plot.legend(loc="lower right", fontsize="small")
# plot.show()
