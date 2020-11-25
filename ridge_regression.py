import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Drop rows after 01/01/2017
df = df[df["Timestamp"] > timestamp]

# Create new column with python datetime to plt graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)


# Setting hyperparameters to try with Ridge regression
parameters = {"alpha": [1],
              "solver": ["sag", "saga"]}
# Using GridSearchCV to optimise model
regressor = GridSearchCV(Ridge(), parameters)
regressor.fit(scaled_x_train, scaled_y_train)
# Evaluating model score and predicting values
y_predicted = regressor.predict(scaled_x_test)
score = regressor.score(scaled_x_test, scaled_y_test)

print("R^2 Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))
print("Best params: " + str(regressor.best_params_))

plot.scatter(x_test.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Predicted R^2 Score: " + str(format(score, ".3f")),
             s=1)
# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], minMaxScaler.inverse_transform(scaled_y_test[::50]),
             s=1, label="Actual")
plot.title("Bitcoin price in USD alongside predicted price using Ridge regression")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()
