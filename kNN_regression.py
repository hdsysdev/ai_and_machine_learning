# Author: Hubert Dudowicz - S17119577
# n this file a kNN model is trained and scored. Price data is predicted based on the dates in the testing set.
# The actual testing data is then plotted against the predicted price data.

# Import required libraries
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Get dataframe dropping rows before 01/01/2017
df = df[(df["Timestamp"] > timestamp)]

# Get X and Y axes for training the dataset
x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

# Split data into 2/3 training and 1/3 testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Create Min Max Scaler to scale data prior to training
minMaxScaler = MinMaxScaler()

# Scale training data using min max scaler
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
# Scale testing data using min max scaler
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Creating and training KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(scaled_x_train, scaled_y_train)
# Scoring model on testing data
score = model.score(scaled_x_test, scaled_y_test)
# Predicting price using model to plot against the original price
y_predicted = model.predict(scaled_x_test)

# Print score
print("Score: " + str(score))
# Calculate and display mean squared error using actual test set price data and scaled price data predicted by the model
# inversely transformed into the original scale it was prior to scaling
print("Mean Squared Error: " + str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))

# Converting timestamp values to datetime64 for plotting as human readable time
# Plot original testing values to compare to predicted values
plot.scatter(x_test.values.astype(dtype='datetime64[s]'),
             y_test,
             s=1, label="Actual")
# Plot predicted values, inversely transforming them to the original scale using the min max scaler
plot.scatter(x_test.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label=" R^2 Score: " + str(format(score, ".5f")),
             s=1)
# Set title and axis labels for graph
plot.title("kNN predicted bitcoin USD price on testing data")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
# Show legend and grid
plot.grid(True)
plot.legend(loc="lower right", fontsize="small")
# Show plotted graph
plot.show()

