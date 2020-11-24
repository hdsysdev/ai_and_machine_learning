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
# Drop rows after 01/01/2017
df = df[(df["Timestamp"] > timestamp)]

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()

scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Training KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(scaled_x_train, scaled_y_train)
# Scoring model on testing data and predicting price to plot against original price
score = model.score(scaled_x_test, scaled_y_test)
y_predicted = model.predict(scaled_x_test)
print("Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))

# Plot training values to compare to prediction
plot.scatter(x_test.values.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(scaled_y_test),
             s=1, label="Actual")
# Plot predicted values
plot.scatter(x_test.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label=" R^2 Score: " + str(format(score, ".5f")),
             s=1)
# Set title and axis labels for graph
plot.title("kNN predicted bitcoin USD price on testing data")
plot.ylabel("Predicted Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()

