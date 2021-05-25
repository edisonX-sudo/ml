import numpy as np
import numpy as py
import matplotlib.pyplot as plt

'''
    f(x) = theta0 + theta1*x1 + theta2*x2 ...thetan*xn
    E(theta) = 0.5 * sigma(i=1,i<n)(ftheta(xi)-yi)**2 // dataCount as n
    thetaj = thetaj - ETA * sigma(i=1,i<n)(ftheta(xi)-yi)*xj // 0<=j<=n
'''


def standardize(x: np.ndarray):
    return (x - x.mean()) / x.std()


thetas = np.random.rand(3)


def toVarMatrix(x: np.ndarray):
    return np.vstack((x ** 0, x, x ** 2)).T


def f(x):
    return np.dot(toVarMatrix(x), thetas)


def MSE(x: np.ndarray, y: np.ndarray):
    return 1 / x.shape[0] * (0.5 * np.sum((f(x) - y) ** 2))


if __name__ == '__main__':
    train = py.loadtxt('train.cvs', delimiter=',', skiprows=1)
    t_x = train[:, 0]
    t_y = train[:, 1]
    t_z = standardize(t_x)

    ETA = 1e-2
    diff = 1
    count = 0
    err = [MSE(t_z, t_y)]
    while diff > 1e-2:
        thetas = thetas - ETA * np.dot((f(t_z) - t_y), toVarMatrix(t_z))
        err.append(MSE(t_z, t_y))
        diff = err[-2] - err[-1]
        count += 1
        print('diff:{:2f},count:{},err:{:4f},thetas:{}'.format(diff, count, err[-1], thetas))

    x = np.linspace(t_z.min(), t_z.max(), 1000)
    plt.plot(t_z, t_y, 'o')
    plt.plot(x, f(x))
    plt.show()
    countX = np.linspace(0, len(err), len(err))
    plt.plot(countX,err)
    plt.show()
