# In this file polynomial features are generated to various degrees and a linear regression model is trained
# for each to find the most effective degree. Predicted data is plotted against all the actual data after
# 2017 for comparison.

# Import required libraries
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

# Create new column with python datetime to plot graph
df["Date"] = df["Timestamp"].values.astype(dtype='datetime64[s]')

# Get X and Y axes for training the dataset
x = pandas.DataFrame(df["Timestamp"])
y = pandas.DataFrame(df["Close"])

# Split training and test sets from full set of data with 33% going to testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Create Min Max Scaler to scale data prior to training
minMaxScaler = MinMaxScaler()
# Scale training data using min max scaler
scaled_x_train = minMaxScaler.fit_transform(x_train)
scaled_y_train = minMaxScaler.fit_transform(y_train)
# Scale testing data using min max scaler
scaled_x_test = minMaxScaler.fit_transform(x_test)
scaled_y_test = minMaxScaler.fit_transform(y_test)

# Plotting original test values to compare to predicted values
plot.scatter(x_test.values.astype(dtype='datetime64[s]')[::50],
             y_test[::50], s=1, label="Regular")
# Generating polynomial features up to a degree of 8 and training models to find the most accurate degree
for polyDegree in range(2, 9):
    # Creating a pipeline generating the current degree of polynomial features then training linear regression model
    # on training data
    lr = make_pipeline(PolynomialFeatures(polyDegree), LinearRegression())
    # Fitting model on training data
    lr.fit(scaled_x_train, scaled_y_train)
    # Predicting price based on test set
    y_predicted = lr.predict(scaled_x_test)
    # Calculating R^2 score against testing sets
    score = lr.score(scaled_x_test, scaled_y_test)
    # Plotting predictions made by linear regression with the current degree of polynomical features
    plot.scatter(x_test.values.astype(dtype='datetime64[s]'),
                 minMaxScaler.inverse_transform(y_predicted),
                 label=str(polyDegree) + "° R^2: " + str(format(score, ".3f")), s=5)
    # Print score for current model
    print("Degree " + str(polyDegree) + " Score: " + str(score))
    # Calculate and display mean squared error for the current model using test set price data and scaled price data
    # predicted by the model inversely transformed into the original scale it was prior to scaling
    print("Degree " + str(polyDegree) + " Mean Squared Error: " +
          str(mean_squared_error(y_test, minMaxScaler.inverse_transform(y_predicted))))

# Set title and axis labels for graph
plot.title("Polynomial regression to predict future price of bitcoin in USD")
plot.ylabel("Closing Price in $")
plot.xlabel("Date")
plot.xticks(rotation=40)
# Show legend and grid
plot.grid(True)
plot.legend(loc="upper left", fontsize="small")
# Show graph
plot.show()
