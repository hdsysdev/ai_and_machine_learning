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

# plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
# plot.title("Bitcoin closing price from the start of 2017")
# plot.ylabel("Closing Price in $")
# plot.xlabel("Date")
# plot.xticks(rotation=40)
# plot.grid(True)
# plot.show()

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], scaled_y_test[::50], s=1, label="Regular")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
for polyDegree in range(2, 9):
    lr = make_pipeline(PolynomialFeatures(polyDegree), LinearRegression())
    lr.fit(scaled_x_train, scaled_y_train)
    y_predicted = lr.predict(scaled_x_test)
    score = lr.score(scaled_x_test, scaled_y_test)
    # Using scatter plot as plot_date function's linewidth property isn't working
    plot.scatter(x_test.values.astype(dtype='datetime64[s]'),
                 y_predicted, label="Degree " + str(polyDegree) + " R^2 Score: " + str(format(score, ".3f")),
                 s=12)
    print("Polynomial Degree " + str(polyDegree) + " Score: " + str(score))

plot.legend(loc="lower right", fontsize="small")
plot.show()
