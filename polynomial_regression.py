import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

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

x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
minMaxScaler = MinMaxScaler()
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_train = minMaxScaler.fit_transform(y_train)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Plotting original test values to compare to predicted values
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50],
             scaled_y_test[::50], s=1, label="Regular")
# Generating polynomial features up to a degree of 8 to find the most optimal degree
for polyDegree in range(2, 9):
    lr = make_pipeline(PolynomialFeatures(polyDegree), LinearRegression())
    # Fitting model on training data
    lr.fit(scaled_x_train, scaled_y_train)
    y_predicted = lr.predict(scaled_x_test)
    score = lr.score(scaled_x_test, scaled_y_test)
    # Plotting predictions made by linear regression with polynomical features
    plot.scatter(x_test.values.astype(dtype='datetime64[s]'),
                 y_predicted, label=str(polyDegree) + "° R^2: " + str(format(score, ".3f")),
                 s=5)
    print("Degree " + str(polyDegree) + "Score: " + str(score))
    print("Degree " + str(polyDegree) + " Mean Squared Error: " +
          str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))

plot.title("Polynomial regression to predict future price of bitcoin in USD")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.grid(True)
plot.xticks(rotation=40)
plot.legend(loc="upper left", fontsize="small")
plot.show()