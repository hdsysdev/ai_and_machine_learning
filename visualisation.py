import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Create new column with python datetime to plot graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

# Drop rows after 01/01/2017
dfFrom2017 = df[df["Timestamp"] > timestamp]
# Plot graph using matplotlib
plot.plot_date(x=dfFrom2017["Date"], y=dfFrom2017["Close"], fmt="b")
plot.title("Bitcoin closing price from January 2017")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()

# Plot graph using matplotlib since start of data
plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
plot.title("Bitcoin closing price from January 2012")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()

