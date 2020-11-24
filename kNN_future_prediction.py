import joblib
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

df_train = df[(df["Timestamp"] >= timestamp) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2019").timestamp())]

df_validate = df[(df["Timestamp"] >= timestamp)]
validate_x = df_validate[["Timestamp"]]
validate_y = df_validate[["Close"]]
# Create new column with python datetime to plt graph
df_train["Date"] = df_train["Timestamp"].values.astype(dtype='datetime64[s]')

test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2019").timestamp())]
test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

x = pandas.DataFrame(df_train["Timestamp"])
y = pandas.DataFrame(df_train["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

minMaxScaler = MinMaxScaler()

scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)

scaled_test_x = minMaxScaler.fit_transform(test_x)
scaled_test_y = minMaxScaler.fit_transform(test_y)
model = KNeighborsRegressor()
model.fit(scaled_x_train, scaled_y_train)
y_predicted = model.predict(scaled_test_x)

# Calculating score and mean squared error
score = model.score(scaled_test_x, scaled_test_y)
print("Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(test_y, y_predicted)))

plot.scatter(validate_x.values.astype(dtype='datetime64[s]')[::50],
             validate_y[::50], s=1, label="Original")
plot.scatter(test_x.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Predicted",
             s=1)
plot.title("Bitcoin price in USD alongside predicted price using kNN")
plot.ylabel("Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()


