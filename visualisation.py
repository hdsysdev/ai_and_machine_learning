# Author: Hubert Dudowicz - S17119577
# In this file the dataset is visualised using pyplot. Price data since the start of the dataset is plotted on one graph
# as well as data after 2017 being plotted in another.

# Import required libraries
import matplotlib.pyplot as plot
import pandas

# Read CSV to pandas dataframe
data = pandas.read_csv("bitstamp.csv")
# Create new dataframe dropping rows with NaN values
df = data.dropna().copy()

# Get unix timestamp for 01/01/2017
timestamp = pandas.Timestamp("01/01/2017").timestamp()
# Create new date column with python datetime to plot graph with dates on the x axis
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

# Drop rows after 01/01/2017
dfFrom2017 = df[df["Timestamp"] > timestamp]
# Plot graph of data after 2017 using matplotlib.
plot.plot_date(x=dfFrom2017["Date"], y=dfFrom2017["Close"], fmt="b")
# Set graph and label titles. Turn on the grid. Rotate labels for visibility
plot.title("Bitcoin closing price from January 2017")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()

# Plot graph using matplotlib since start of dataset
plot.plot_date(x=df["Date"], y=df["Close"], fmt="b")
# Set graph and label titles. Turn on the grid. Rotate labels for visibility
plot.title("Bitcoin closing price from January 2012")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
plot.grid(True)
plot.show()

