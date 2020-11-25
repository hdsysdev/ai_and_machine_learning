import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

# Split training and test sets from full set of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Training model on training data
lr = LinearRegression()
lr.fit(scaled_x_train, scaled_y_train)

# Scoring model and predicting price based on test set to calculate MSE.
y_predicted = lr.predict(scaled_x_test)
score = lr.score(scaled_x_test, scaled_y_test)
# Plotting actual price of bitcoin from test set. Using every 50th value to avoid over-congestion
plot.scatter(x_test.astype(dtype='datetime64[s]')[::50],
             y_test[::50],
             s=1.2, label="Actual")
# Plotting predicted price of bitcoin. Using MinMaxScaler to transform the values back to USD
plot.scatter(x_test.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Predicted R^2 Score: " + str(format(score, ".3f")),
             s=1.5)
plot.title("Linear regression predicted price of bitcoin in USD")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()
print("Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))
