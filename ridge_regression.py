import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC, SVR

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


x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)
param_grid = {'alpha': [10, 8, 6, 4, 2, 1.0, 0.75, 0.5, 0.25, 0.2, 0.1]}
model = make_pipeline(PolynomialFeatures(5), GridSearchCV(Ridge(), param_grid=param_grid))

model.fit(scaled_x_train, scaled_y_train.ravel())
print(model.score(scaled_x_test, scaled_y_test))
