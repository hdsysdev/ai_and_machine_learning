import joblib
import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Drop rows after 01/01/2017
df = df[df["Timestamp"] > timestamp]

# Create new column with python datetime to plt graph
# df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')
#
# plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
# plot.title("Bitcoin closing price from the start of 2017")
# plot.ylabel("Closing Price in $")
# plot.xlabel("Date")
# plot.xticks(rotation=40)
# plot.grid(True)
# plot.show()
test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2017").timestamp()) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2018").timestamp())]

test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()

scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_x_test = minMaxScaler.fit_transform(test_x)
scaled_y_test = minMaxScaler.fit_transform(test_y)
model = KNeighborsRegressor(n_neighbors=100, weights='uniform')

model.fit(scaled_x_train, scaled_y_train)

plot.scatter(test_x.values.astype(dtype='datetime64[s]')[::50], minMaxScaler.inverse_transform(scaled_y_test[::50]),
             s=1, label="Regular")
y_predicted = model.predict(scaled_x_test)
score = model.score(scaled_x_test, scaled_y_test)
# Using scatter plot as plot_date function's linewidth property isn't working
plot.scatter(test_x.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label=" R^2 Score: " + str(format(score, ".3f")),
             s=12)
plot.legend(loc="lower right", fontsize="small")
plot.show()
print("Score: " + str(score))