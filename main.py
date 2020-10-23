import pandas
import numpy as np
import matplotlib.pyplot as plt
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

plt.plot_date(x=df["Date"], y=df["Close"], fmt="b")
plt.title("Bitcoin price from the start of 2017")
plt.ylabel("Price in $")
plt.grid(True)
plt.show()
# var = data.groupby("Close")
