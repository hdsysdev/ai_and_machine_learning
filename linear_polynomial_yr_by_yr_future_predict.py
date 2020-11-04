import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
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
# Drop rows after 01/01/2017
df_train = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2018").timestamp()) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2019").timestamp())]

# Create new column with python datetime to plt graph
df_train["Date"] = df_train["Timestamp"].values.astype(dtype='datetime64[s]')

# plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
# plot.title("Bitcoin closing price from the start of 2017")
# plot.ylabel("Closing Price in $")
# plot.xlabel("Date")
# plot.xticks(rotation=40)
# plot.grid(True)
# plot.show()

test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2019").timestamp())]
print(test_df)
test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

x = pandas.DataFrame(df_train["Timestamp"])
y = pandas.DataFrame(df_train["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)

scaled_test_x = minMaxScaler.fit_transform(test_x)
scaled_test_y = minMaxScaler.fit_transform(test_y)


# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], scaled_y_test[::50], s=1, label="Train")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
lr = make_pipeline(PolynomialFeatures(), LinearRegression())
lr.fit(scaled_x_train, scaled_y_train)
y_predicted = lr.predict(scaled_test_x)
score = lr.score(scaled_test_x, scaled_test_y)
# Using scatter plot as plot_date function's linewidth property isn't working
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             scaled_test_y, label="Actual",
             s=12)
plot.scatter(test_x.values.astype(dtype='datetime64[s]'),
             y_predicted, label="Degree 8" + " R^2 Score: " + str(format(score, ".3f")),
             s=12)
print("Polynomial Degree 8" + " Score: " + str(score))

plot.legend(loc="lower right", fontsize="small")
plot.show()
