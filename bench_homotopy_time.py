import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
random_state = 414


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()


def homotopy_path(X, Y, lambda_, coef, y_t, epsilon=1e-3, nu=1.):

    eps_0 = epsilon / 10.
    step_size = np.sqrt(2. * (epsilon - eps_0) / nu)
    Y_t = np.array(list(Y[:-1]) + [y_t], order='F')
    y_stop = Y[-1]

    while y_t < y_stop:

        y_t = min(y_t + step_size, y_stop)
        Y_t[-1] = y_t
        tol = eps_0 / np.linalg.norm(Y_t) ** 2
        alpha = [lambda_ / X.shape[0]]
        res = lasso_path(X, Y_t, alphas=alpha, coef_init=coef, eps=tol)
        coef = res[1].ravel()

    while y_t > y_stop:

        y_t = max(y_t - step_size, y_stop)
        Y_t[-1] = y_t
        tol = eps_0 / np.linalg.norm(Y_t) ** 2
        alpha = [lambda_ / X.shape[0]]
        res = lasso_path(X, Y_t, alphas=alpha, coef_init=coef, eps=tol)
        coef = res[1].ravel()

    return coef


if __name__ == '__main__':

    import time
    # X = np.load("Xclimate.npy")
    # Y = np.load("yclimate.npy")

    n_samples, n_features = (1000, 5000)
    X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                           random_state=random_state)

    X /= np.linalg.norm(X, axis=0)
    mask = np.sum(np.isnan(X), axis=0) == 0
    if np.any(mask):
        X = X[:, mask]

    Y = (Y - Y.mean()) / Y.std()
    Y /= np.linalg.norm(Y)
    print("norm(Y) = ", np.linalg.norm(Y))
    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)
    lambda_ = np.linalg.norm(X.T.dot(Y), ord=np.inf) / 20.

    methods = [r"Homotopy", r"CD on $\mathcal{D}_{n+1}(y_{n+1})$",
               r"CD initialized with $\beta(\mathcal{D}_n)$"]
    method_colors = ["b", "g", "r"]

    epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
    len_eps = len(epsilons)
    times = np.zeros((len_eps, len(methods)))

    for i_eps, eps in enumerate(epsilons):

        print("\n eps is", eps)

        # Initial fitting
        alpha = [lambda_ / X[:-1].shape[0]]
        tic = time.time()
        eps_ = eps / np.linalg.norm(Y[:-1]) ** 2
        coef = lasso_path(X[:-1], Y[:-1], alphas=alpha, tol=eps_)[1].ravel()
        t_init = time.time() - tic
        print("t_init =", t_init)
        y_t = X[-1:].dot(coef)[0]

        # Compute solution with homotopy
        coef_c = coef.copy()
        tic = time.time()
        coef_hom = homotopy_path(X, Y, lambda_, coef_c, y_t=y_t, epsilon=eps)
        t_homotopy = time.time() - tic
        print("t_homotopy =", t_homotopy)

        # Compute full solution without init
        alpha = [lambda_ / X.shape[0]]
        tic = time.time()
        beta = lasso_path(X, Y, alphas=alpha, tol=eps_)[1]
        t_full_woinit = time.time() - tic
        print("t_full_woinit =", t_full_woinit)

        # Compute full with init
        alpha = [lambda_ / X.shape[0]]
        eps_ = eps / np.linalg.norm(Y) ** 2
        tic = time.time()
        beta = lasso_path(X, Y, alphas=alpha, coef_init=coef, tol=eps_)[1]
        t_full_init = time.time() - tic
        print("t_full_init =", t_full_init)

        times[i_eps] = t_homotopy, t_full_woinit, t_full_init

np.save("bench_homotopy.npy", times)

df = pd.DataFrame(times, columns=methods)

fig, ax = plt.subplots(1, 1)
df.plot(kind='bar', ax=ax, rot=0, color=method_colors)
x_eps = np.log10(epsilons).astype(np.intc)
plt.xticks(range(len_eps), [r"$10^{%s}$" % (np.str(t)) for t in x_eps])
plt.xlabel("Duality gap")
plt.ylabel("Time (s)")
plt.grid(color='w')
leg = plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("homotopy_time.pdf", format="pdf")
plt.show()
