import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Drop rows after 01/01/2017
df = df[df["Timestamp"] > timestamp]
print(df)
# var = data.groupby("Close")
