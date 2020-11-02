import pandas
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
from sklearn.covariance import GraphicalLasso
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVR

# Read CSV to pandas dataframe
from sklearn.tree import DecisionTreeRegressor

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

test_df = df[(df["Timestamp"] >= pandas.Timestamp("01/01/2017").timestamp()) & (
            df["Timestamp"] <= pandas.Timestamp("01/01/2018").timestamp())]
x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

test_x = test_df[["Timestamp"]]
test_y = test_df[["Close"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)

# scaled_x_test = minMaxScaler.fit_transform(test_x)
# scaled_y_test = minMaxScaler.fit_transform(test_y)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Converting timestamp values to datetime64 for plotting as human readable time
# Plotting every 50th value to avoid over-congestion of points
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50], minMaxScaler.inverse_transform(scaled_y_test[::50]),
             s=1, label="Regular")
Ridge().get_params().keys()
# Generating polynomial features up to a degree of 8 to find the most optimal degree
parameters = {"fit_intercept": [True, False],
              "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}

regressor = GridSearchCV(Ridge(), parameters, verbose=1)
regressor.fit(scaled_x_train, scaled_y_train)

y_predicted = regressor.predict(scaled_x_test)
score = regressor.score(scaled_x_test, scaled_y_test)
# Using scatter plot as plot_date function's linewidth property isn't working
plot.scatter(x_test.astype(dtype='datetime64[s]'),
             minMaxScaler.inverse_transform(y_predicted), label="Degree 6" + " R^2 Score: " + str(format(score, ".3f")),
             s=12)
print("Polynomial Degree 6" + " Score: " + str(score))

plot.legend(loc="lower right", fontsize="small")
plot.show()
