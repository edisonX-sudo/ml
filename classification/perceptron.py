import numpy as np
import matplotlib.pyplot as plt

w = np.random.rand(2)


def f(x):
    if np.dot(w, x) > 0:
        return 1
    else:
        return -1


if __name__ == '__main__':
    train = np.loadtxt('image1.cvs', delimiter=',', skiprows=1)
    train_x = train[:, 0:2]
    train_y = train[:, 2]

    while True:
        arg_changed = False

        for x, y in zip(train_x, train_y):
            if f(x) != y:
                w = w + y * x
                arg_changed = True

        if not arg_changed:
            break

    x1 = np.arange(0, 500)
    plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'x')
    plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'o')
    plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
    plt.axis('scaled')
    plt.show()
