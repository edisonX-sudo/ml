import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def standardize(x: np.ndarray):
    return (x - x.mean()) / x.std()


def f(x, regression):
    return regression.coef_[0][0] * x + regression.coef_[0][1] * x ** 2 + regression.intercept_[0]


if __name__ == '__main__':
    train_cvs = pd.read_csv('train.cvs')
    train_cvs.sort_values('x', inplace=True)
    x_train = train_cvs[['x']]
    y_train = train_cvs[['y']]

    #x_train, x_test, y_train, y_test = train_test_split(x_tc, y_tc, random_state=1)

    lreg = LinearRegression()
    lreg.fit(x_train, y_train)

    plt.plot(x_train, y_train, 'b.')
    plt.plot(x_train, lreg.predict(x_train), 'r')

    preg = PolynomialFeatures(degree=2)
    x_p_train = preg.fit_transform(x_train)

    lreg1 = LinearRegression()
    lreg1.fit(x_p_train, y_train)

    plt.plot(x_train, lreg1.predict(x_p_train), 'y')
    plt.show()
