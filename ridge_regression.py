# In this file a ridge regression model is trained and the hyper-parameters are optimised using GridSearchCV.
# Price data is then predicted using the model then this predicted data is plotted against
# all of the actual data after 2017 for comparison.

# Import required libraries
import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Drop rows after 01/01/2017
df = df[df["Timestamp"] > timestamp]

# Create new column with python datetime to plot graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

# Get X and Y axes for training the dataset
x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

# Split training and test sets from full set of data with 33% going to testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Create Min Max Scaler to scale data prior to training
minMaxScaler = MinMaxScaler()
# Scale training data using min max scaler
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
# Scale testing data using min max scaler
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)


# Setting hyperparameters to try optimise Ridge regression
parameters = {"alpha": [1, 2, 5, 10],
              "solver": ["svd", "sparse_cg", "sag", "saga"]}
# Creating Ridge regression model with GridSearchCV to optimise model
regressor = GridSearchCV(Ridge(), parameters)
regressor.fit(scaled_x_train, scaled_y_train)

# Evaluating model score and predicting values
y_predicted = regressor.predict(scaled_x_test)
score = regressor.score(scaled_x_test, scaled_y_test)

# Print score
print("R^2 Score: " + str(score))
# Calculate and display mean squared error using actual test set price data and scaled price data predicted by the model
# inversely transformed into the original scale it was prior to scaling
print("Mean Squared Error: " + str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))
# Printing best parameters found by grid search
print("Best params: " + str(regressor.best_params_))

# Plotting predicted price to compare to actual price
plot.scatter(x_test.astype(dtype='datetime64[s]'), minMaxScaler.inverse_transform(y_predicted),
             label="Predicted R^2 Score: " + str(format(score, ".3f")), s=1)
# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th testing value to avoid over-congestion of points
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], minMaxScaler.inverse_transform(scaled_y_test[::50]),
             s=1, label="Actual")

# Set title and axis labels for graph
plot.title("Bitcoin price in USD alongside predicted price using Ridge regression")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
# Show legend and grid then display graph
plot.grid(True)
plot.legend(loc="lower right", fontsize="small")
plot.show()
