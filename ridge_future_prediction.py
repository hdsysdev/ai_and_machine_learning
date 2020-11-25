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
# Drop rows after 01/01/2017 and after 01/01/2020
df_train = df[(df["Timestamp"] >= timestamp) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2020").timestamp())]

# Create new column with python datetime to plt graph
df_train["Date"] = df_train["Timestamp"].values.astype(dtype='datetime64[s]')

# Getting test data after 01/01/2020
test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2020").timestamp())]
test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

x = pandas.DataFrame(df_train["Timestamp"])
y = pandas.DataFrame(df_train["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_x_test = minMaxScaler.fit_transform(test_x)
scaled_y_test = minMaxScaler.fit_transform(test_y)

# Using GridSearchCV to optimise model
regressor = Ridge(alpha=1, solver="sag")
regressor.fit(scaled_x_train, scaled_y_train)
y_predicted = regressor.predict(scaled_x_test)
score = regressor.score(scaled_x_test, scaled_y_test)
# Using scatter plot as plot_date function's linewidth property isn't working
print("R^2 Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(test_y, minMaxScaler.inverse_transform(y_predicted))))

# Plotting training values
plot.scatter(x_train.astype(dtype='datetime64[s]'),
             y_train, label="Train",
             s=1)
# Plotting predicted values
plot.scatter(test_x.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Predicted R^2 Score: " + str(format(score, ".3f")),
             s=1)
# Plotting test values
plot.scatter(test_x.values.astype(dtype='datetime64[s]')[::50],
             minMaxScaler.inverse_transform(scaled_y_test[::50]),
             s=1, label="Actual")
plot.title("Bitcoin price in USD alongside predicted price using Ridge regression")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()
