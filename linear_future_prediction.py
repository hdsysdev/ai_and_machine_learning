import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import LinearSVC

# Read CSV to pandas dataframe
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
scaled_test_x = minMaxScaler.fit_transform(test_x)
scaled_test_y = minMaxScaler.fit_transform(test_y)

# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
plot.scatter(x_train.values.astype(dtype='datetime64[s]')[::50],
             y_train[::50], s=1, label="Train")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
lr = LinearRegression()
lr.fit(scaled_x_train, scaled_y_train)
y_predicted = lr.predict(scaled_test_x)
score = lr.score(scaled_test_x, scaled_test_y)
# Plotting test data after 01/01/20
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(scaled_test_y), label="Testing",
             s=1)
# Plotting predicted price
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="R^2 Score: " + str(format(score, ".3f")),
             s=2)

print("Score: " + str(score))
print("Mean Squared Error: " + str(mean_squared_error(test_y, minMaxScaler.inverse_transform(y_predicted))))

plot.title("Linear regression to predict future price of bitcoin in USD")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="lower right", fontsize="small")
plot.show()