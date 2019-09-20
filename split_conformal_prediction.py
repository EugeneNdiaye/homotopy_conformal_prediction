import numpy as np
from sklearn.model_selection import train_test_split
from tools import logcosh_reg, linex_reg
import intervals
from sklearn.datasets import make_regression
from sklearn.linear_model import lasso_path
random_state = 414


def conf_pred(X, Y_seen, lambda_, Y_range, alpha=0.1, method="lasso"):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X[:-1, :], Y_seen, test_size=0.5, random_state=414)

    # Training
    if method is "lasso":
        lmd = [lambda_ / X_train.shape[0]]
        res = lasso_path(X_train, Y_train, alphas=lmd, eps=1e-12)
        coef = res[1].ravel()

    elif method is "logcosh":
        coef = logcosh_reg(X_train, Y_train, lambda_)

    elif method is "linex":
        coef = linex_reg(X_train, Y_train, lambda_)

    # Ranking on the test
    mu = X_test.dot(coef)
    sorted_residual = np.sort(np.abs(Y_test - mu))
    index = int((X.shape[0] / 2 + 1) * (1 - alpha))
    quantile = sorted_residual[index]

    mu = X[-1, :].dot(coef)

    return intervals.closed(mu - quantile, mu + quantile)


if __name__ == '__main__':

    alpha = 0.1
    n_samples, n_features = (300, 100)
    X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)
    Y = (Y - Y.mean()) / Y.std()
    lambda_ = np.sqrt(np.log(n_features)) / n_samples
    Y_range = np.min(Y[:-1]), np.max(Y[:-1])
    epsilon = 1e-6 * np.linalg.norm(Y[:-1]) ** 2
    print(epsilon)

    pred_set = conf_pred(X, Y[:-1], lambda_, Y_range, alpha)
    print(str(Y[-1]) + " in", pred_set, Y[-1] in pred_set)
