# Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn import preprocessing
import sys


def interval(values):
    """
    Function takes the time interval and transforms it in a time points array
    :param values: the interval of time
    :return: array of time points
    """
    splitted = values[0].split('-')
    begin = int(splitted[0])
    end = int(splitted[1])
    time_points = [begin]

    i = 0;
    while i<4:
        i +=1
        time_points.append(begin+i)

    return time_points


def array_to_column(time_points):
    """
    Function transforms an array to a column
    :param time_points: array
    :return: column of that array
    """
    column = []
    for time_point in time_points:
        column.append([time_point])
    return np.array(column)


def linear_prediction(file_name, columns, values, separator, x_col, prediction):
    plt.clf()

    if "-" in values[0]:
        values = interval(values)
        values_2d = array_to_column(values)
    else:
        values[0] = int(values[0])
        values_2d = [values]

    # values_2d = [["20210316"]]

    style.use("ggplot")

    data = pd.read_csv(file_name, sep=separator)

    le = preprocessing.LabelEncoder()
    # encode columns in case they are not numeric
    column1 = le.fit_transform(list(data[columns[0]]))
    column2 = le.fit_transform(list(data[columns[1]]))

    predict = prediction

    data = data[columns]

    # data = shuffle(data) # Optional - shuffle the data

    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
    best = 0
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

    x_test2 = np.concatenate([np.array(x_test), np.array(values_2d)])

    start = len(x_test)
    end = len(x_test2)
    predicted = linear.predict(x_test2)

    x = len(predicted)
    predicted_val = predicted[start:end]

    return predicted_val


####################################################################################
file = "E:\\MAP\\BitcoinSpring\\BitcoinTrader\\CSV\\filecoins.csv"
sep = ","
x_col = "Date"
pred = "Price"
my_columns = ["Date", "Price"]
my_values = [sys.argv[1]]

time_points = interval(my_values)
array_to_column(time_points)

prediction = linear_prediction(file, my_columns, my_values, sep, x_col, pred)
for pred in prediction:
    print(round(pred,2))
