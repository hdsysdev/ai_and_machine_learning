# Author: Hubert Dudowicz - S17119577
# In this file a linear regression model is trained on past data from 2017 to 2020. Price data after 2020 is predicted
# using the model then this predicted data is plotted against all of the actual data after 2017 for comparison.

# Import required libraries
import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th training value to avoid over-congestion of points
plot.scatter(x_train.values.astype(dtype='datetime64[s]')[::50],
             y_train[::50], s=1, label="Train")

# Creating and training Linear Regression model
lr = LinearRegression()
lr.fit(scaled_x_train, scaled_y_train)
# Predicting price using model to plot against the original price
y_predicted = lr.predict(scaled_test_x)
# Scoring model on testing data
score = lr.score(scaled_test_x, scaled_test_y)
# Converting timestamp values to datetime64 for plotting as human readable time
# PLot original testing price data after 01/01/2020.
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             test_y, label="Testing",
             s=1)
# Plotting predicted price to compare to actual price
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="R^2 Score: " + str(format(score, ".3f")),
             s=2)

# Print score
print("Score: " + str(score))
# Calculate and display mean squared error using actual test set price data and scaled price data predicted by the model
# inversely transformed into the original scale it was prior to scaling
print("Mean Squared Error: " + str(mean_squared_error(test_y, minMaxScaler.inverse_transform(y_predicted))))

# Set title and axis labels for graph
plot.title("Linear regression to predict future price of bitcoin in USD")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
# Show legend and grid
plot.grid(True)
plot.legend(loc="lower right", fontsize="small")
# Show plotted graph
plot.show()
