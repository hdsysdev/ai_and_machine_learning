# Author: Hubert Dudowicz - S17119577
# In this file a kNN model is trained on past data from 2017 to 2020. Price data after 2020 is predicted using the model
# then this predicted data is plotted against all of the actual data after 2017 for comparison.

# Import required libraries
import matplotlib.pyplot as plot
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()

# Get dataframe containing entries between 01/01/2017 and 01/01/2020
df_train = df[(df["Timestamp"] >= timestamp) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2020").timestamp())]

# Create validation set of entries after 01/01/2017
df_validate = df[(df["Timestamp"] >= timestamp)]
validate_x = df_validate[["Timestamp"]]
validate_y = df_validate[["Close"]]

# Create new column from timestamp with python datetime to plot graph with dates on the x axis
df_train["Date"] = df_train["Timestamp"].values.astype(dtype='datetime64[s]')

# Create dataframe from entries after 01/01/2020 to test trained model on the unseen future data
test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2020").timestamp())]
# Create sets from testing dataframe to use in testing the model
test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

# Get X and Y axes for training the dataset
x_train = pandas.DataFrame(df_train["Timestamp"])
y_train = pandas.DataFrame(df_train["Close"])

# Create Min Max Scaler to scale data prior to training
minMaxScaler = MinMaxScaler()
# Scale training data using min max scaler
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
# Scale testing data using min max scaler
scaled_test_x = minMaxScaler.fit_transform(test_x)
scaled_test_y = minMaxScaler.fit_transform(test_y)

# Create and fit k nearest neighbors model on scaled training data
model = KNeighborsRegressor()
model.fit(scaled_x_train, scaled_y_train)
# Create predicted price data using trained model
y_predicted = model.predict(scaled_test_x)

# Calculating score based on testing data
score = model.score(scaled_test_x, scaled_test_y)
# Print score
print("Score: " + str(score))
# Calculate and display mean squared error using actual price data and price data predicted by the model
print("Mean Squared Error: " + str(mean_squared_error(test_y, y_predicted)))

# Converting timestamp values to datetime64 for plotting as human readable time
# PLot original price data after 01/01/2017. Plotting every 50th entry to avoid over-congestion of points
plot.scatter(validate_x.values.astype(dtype='datetime64[s]')[::50],
             validate_y[::50], s=1, label="Original")
# Plot predicted price data to compare against actual price data
plot.scatter(test_x.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Predicted",
             s=1)
# Set graph and axis labels
plot.title("Bitcoin price in USD alongside predicted price using kNN")
plot.ylabel("Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
# Show grid and legend
plot.grid(True)
plot.legend(loc="lower right", fontsize="small")
# Show graph
plot.show()


