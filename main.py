import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Drop rows after 01/01/2017
df = df[df["Timestamp"] > timestamp]
print(df)

# Create new column with python datetime to plt graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
plot.title("Bitcoin closing price from the start of 2017")
plot.ylabel("Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()
# var = data.groupby("Close")
