import numpy as np
import matplotlib.pyplot as plt

'''
    f(x) = t0*1 + t1*x1 + ... tn*xn
    Et(x) = 0.5 * sigma(i=1;i<=n) (ft(xi)-yi) ** 2 // dateCount as n
    tj = tj - ETA * sigma(i=1;i<=n) (ft(xi)-yi) * xj // 0 <= j <=n 
'''


def standardize(x: np.ndarray):
    return (x - x.mean()) / x.std()


t = np.random.rand(3)


def matrix(x):
    return np.vstack((x ** 0, x, x ** 2)).T


def f(x):
    return np.dot(matrix(x), t)


def MSE(x: np.ndarray, y: np.ndarray):
    return 1 / x.shape[0] * np.sum((f(x) - y) ** 2)


if __name__ == '__main__':
    trainData = np.loadtxt('train.cvs', delimiter=',', skiprows=1)
    t_x = trainData[:, 0]
    t_y = trainData[:, 1]
    plt.plot(t_x, t_y, 'o')
    plt.show()

    t_z: np.ndarray = standardize(t_x)

    ETA = 1e-2
    diff = 1
    err = [MSE(t_z, t_y)]
    count = 0
    while diff > 1e-2:
        p = np.random.permutation(len(t_z))
        for x, y in zip(t_z[p], t_y[p]):
            t = t - ETA * (f(x) - y) * matrix(x)[0]
        err.append(MSE(t_z, t_y))
        diff = err[-2] - err[-1]
        count += 1
        print('diff:{:4f},err:{:4f},count:{},t:{}'.format(diff, err[-1], count, t))

    plt.plot(np.linspace(0, len(err), len(err)), err)
    plt.show()
    t_z_flow = np.linspace(t_z.min(), t_z.max(), 1000)
    plt.plot(t_z_flow, f(t_z_flow))
    plt.plot(t_z, t_y, 'o')
    plt.show()
