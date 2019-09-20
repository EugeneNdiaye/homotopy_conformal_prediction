import numpy as np
from approx_conformal_prediction import conf_pred
import intervals
from sklearn import datasets
from sklearn.model_selection import train_test_split
from split_conformal_prediction import conf_pred as split_conf_pred
from tools import logcosh_reg, linex_reg, cross_val
from sklearn.linear_model import lasso_path
import time
from sklearn.datasets import make_regression

random_state = np.random.randint(100)
print("random_state = ", random_state)

tol_scales = [1e-2, 1e-4, 1e-6, 1e-8]
n_tols = len(tol_scales)
repet = 100
# alpha = 0.1
alpha = 0.9

method = "lasso"
# method = "logcosh"
# method = "linex"

# dataset = "climate"
# dataset = "boston"
dataset = "diabetes"
# dataset = "housing"  # california
# dataset = "synthetic"
# dataset = "friedman1"

if dataset is "boston":
    boston = datasets.load_boston()
    X_full = boston.data
    Y_full = boston.target

if dataset is "diabetes":
    diabetes = datasets.load_diabetes()
    X_full = diabetes.data
    Y_full = diabetes.target

if dataset is "climate":
    X_full = np.load("Xclimate.npy")
    Y_full = np.load("yclimate.npy")

if dataset is "housing":
    housing = datasets.fetch_california_housing()
    X_full, Y_full = housing.data, housing.target

if dataset is "friedman1":
    X_full, Y_full = datasets.make_friedman1(n_samples=500, n_features=50,
                                             noise=0.5)

if dataset is "synthetic":

    n_samples, n_features = (500, 1000)
    X_full, Y_full = make_regression(n_samples=n_samples,
                                     n_features=n_features,
                                     random_state=random_state)


print("Benchmarks on", dataset, "with", method)
max_iter = int(1e8)


# Double check the normalization (excheangeability is preserved but not iid)
# without normalization scipy.minimize fails to converge
X_full /= np.linalg.norm(X_full, axis=0)
mask = np.sum(np.isnan(X_full), axis=0) == 0
if np.any(mask):
    X_full = X_full[:, mask]
Y_full = (Y_full - Y_full.mean()) / Y_full.std()
# normalizing the ovservation to norm 1 is helpfull for convergence of
# scipy.optimize/ So if the message "boom" appears, please, conveniently
# normalize the data with the following line:
# Y_full /= np.linalg.norm(Y_full)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.33, random_state=414)

X_train, X_test = np.asfortranarray(X_train), np.asfortranarray(X_test)
Y_train, Y_test = np.asfortranarray(Y_train), np.asfortranarray(Y_test)

# we select lambda_ by cross validation on the training set
lambda_ = cross_val(X_train, Y_train, method)

res_oracle = np.zeros(3)
res_split = np.zeros(3)
res_approx = np.zeros((3, n_tols))
cov_range = 0

# Conformal prediction set is computed on the test set
print("Number of repetition is ", repet)
random_int = np.arange(Y_test.shape[0])

for i_repet in range(repet):

    # print(i_repet, sep=" ", end=" ")
    print(i_repet)

    np.random.shuffle(random_int)
    X, Y = X_test[random_int, :], Y_test[random_int]
    X_seen = X[:-1, :]
    Y_seen, Y_left = Y[:-1], Y[-1]

    # Range
    Y_range = np.min(Y_seen), np.max(Y_seen)
    cov_range += Y_left in intervals.closed(Y_range[0], Y_range[1])

    # Oracle
    tic = time.time()
    if method is "lasso":
        lmd = [lambda_ / X.shape[0]]
        res = lasso_path(X, Y, alphas=lmd, eps=1e-12, max_iter=max_iter)
        coef_or = res[1].ravel()

    elif method is "logcosh":
        coef_or = logcosh_reg(X, Y, lambda_)

    elif method is "linex":
        coef_or = linex_reg(X, Y, lambda_)

    residual_or = np.abs(Y - X.dot(coef_or))
    q_alpha_or = np.quantile(residual_or, 1 - alpha)
    mu_or = X[-1:].dot(coef_or)[0]
    l_or = mu_or - q_alpha_or
    r_or = mu_or + q_alpha_or
    set_or = intervals.closed(l_or, r_or)
    res_oracle[2] += time.time() - tic
    res_oracle[0] += Y_left in set_or
    res_oracle[1] += r_or - l_or

    # Split
    tic = time.time()
    split_pred_set = split_conf_pred(X, Y_seen, lambda_, Y_range, alpha,
                                     method)
    res_split[2] += time.time() - tic
    res_split[0] += Y_left in split_pred_set
    res_split[1] += split_pred_set.upper - split_pred_set.lower

    # Ridge approximated conformal prediction with different precisions
    for i_tol, tol_scale in enumerate(tol_scales):

        epsilon = tol_scale * np.linalg.norm(Y_seen) ** 2
        # TODO: double check nu
        tic = time.time()
        pred_set = conf_pred(X, Y_seen, lambda_, Y_range, alpha, epsilon,
                             method=method)
        res_approx[2, i_tol] += time.time() - tic
        # We consider the convex hull
        res_approx[0, i_tol] += Y_left in intervals.closed(
            pred_set.lower, pred_set.upper)
        res_approx[1, i_tol] += pred_set.upper - pred_set.lower

res_oracle /= repet
res_split /= repet
res_approx /= repet

print("\n")
print("cov range: \n", cov_range / repet)
print("Oracle: \n", res_oracle)
print("Split: \n", res_split)
print("Approx: \n", res_approx.T)
