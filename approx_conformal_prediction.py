import numpy as np
import intervals
from sklearn.linear_model import lasso_path
from sklearn.linear_model import Ridge
from tools import logcosh_reg, linex_reg


def fit_model(X, Y_t, coef, lambda_, eps_0, method="lasso"):

    if coef is None:
        coef = np.zeros(X.shape[1])

    if method is "lasso":
        tol = eps_0 / np.linalg.norm(Y_t) ** 2
        lmd = [lambda_ / X.shape[0]]
        res = lasso_path(X, Y_t, alphas=lmd, coef_init=coef, eps=tol,
                         max_iter=int(1e8))
        coef = res[1].ravel()

    elif method is "ridge":
        reg = Ridge(alpha=lambda_, fit_intercept=False, solver="auto")
        reg.fit(X, Y_t)
        coef = reg.coef_

    elif method is "logcosh":
        # I cannot early stop scipy.minimize with duality gap :-/
        coef = logcosh_reg(X, Y_t, lambda_, coef)

    elif method is "linex":
        # I cannot early stop scipy.minimize with duality gap :-/
        coef = linex_reg(X, Y_t, lambda_, coef=coef)

    mu = X.dot(coef)

    return mu, np.abs(Y_t - mu), coef


def conf_pred(X, Y_seen, lambda_, y_range, alpha=0.1, epsilon=1e-3, nu=1.,
              method="lasso"):

    pred_set = intervals.empty()
    X = np.asfortranarray(X)
    Y_seen = np.asfortranarray(Y_seen)

    y_min, y_max = y_range
    eps_0 = epsilon / 10. if method is "lasso" else 1e-10
    step_size = np.sqrt(2. * (epsilon - eps_0) / nu)

    # Initial fitting
    coef = fit_model(X[:-1], Y_seen, None, lambda_, eps_0, method)[2]

    y_0 = X[-1:].dot(coef)[0]
    coef_negative_side = coef.copy()
    y_0_negative_side = y_0

    y_t, next_y_t = y_0, y_0
    Y_t = np.array(list(Y_seen) + [y_0], order='F')
    # positive direction
    while next_y_t < y_max:

        next_y_t = min(y_t + step_size, y_max)
        Y_t[-1] = next_y_t
        mu, residual = fit_model(X, Y_t, coef, lambda_, eps_0, method)[:2]
        q_alpha = np.quantile(residual, 1 - alpha)

        y_intv = intervals.closed(y_t, next_y_t)
        mu_intv = intervals.closed(mu[-1] - q_alpha, mu[-1] + q_alpha)
        pred_set = pred_set.union(y_intv.intersection(mu_intv))

        y_t = next_y_t

    coef = coef_negative_side
    y_0 = y_0_negative_side
    y_t, next_y_t = y_0, y_0
    # negative direction
    while next_y_t > y_min:

        next_y_t = max(y_t - step_size, y_min)
        Y_t[-1] = next_y_t
        mu, residual = fit_model(X, Y_t, coef, lambda_, eps_0, method)[:2]
        q_alpha = np.quantile(residual, 1 - alpha)

        y_intv = intervals.closed(next_y_t, y_t)
        mu_intv = intervals.closed(mu[-1] - q_alpha, mu[-1] + q_alpha)
        pred_set = pred_set.union(y_intv.intersection(mu_intv))

        y_t = next_y_t

    return pred_set


if __name__ == '__main__':

    from sklearn.datasets import make_regression
    random_state = 414
    alpha = 0.1
    n_samples, n_features = (30, 20)
    X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)
    Y = (Y - Y.mean()) / Y.std()
    lambda_ = np.sqrt(np.log(n_features)) / n_samples
    Y_range = np.min(Y[:-1]), np.max(Y[:-1])
    epsilon = 1e-2 * np.linalg.norm(Y[:-1]) ** 2
    print(epsilon)

    pred_set = conf_pred(X, Y[:-1], lambda_, Y_range, alpha, epsilon,
                         method="logcosh")
    pred_set = intervals.closed(pred_set.lower, pred_set.upper)
    print(str(Y[-1]) + " in", pred_set, Y[-1] in pred_set)
