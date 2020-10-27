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
df = df[df["Timestamp"] > timestamp]

# Create new column with python datetime to plt graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
plot.title("Bitcoin closing price from the start of 2017")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()


x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)

plot.scatter(x_test.values.astype(dtype='datetime64[s]'), scaled_y_test, s=1, alpha=0.3)
# Generating polynomial features to a degree of 8
for polyDegree in range(2, 9):
    lr = make_pipeline(PolynomialFeatures(polyDegree), LinearRegression())
    lr.fit(scaled_x_train, scaled_y_train)
    y_predicted = lr.predict(scaled_x_test)
    score = lr.score(scaled_x_test, scaled_y_test)
    plot.plot_date(x_test.values.astype(dtype='datetime64[s]'),
                   y_predicted, label="Degree " + str(polyDegree) + " R^2 Score: " + str(score),
                   ls="--")
    print("Polynomial Degree " + str(polyDegree) + " Score: " + str(score))
plot.rcParams['agg.path.chunksize'] = 100000
plot.legend(loc="lower right")
plot.show()
