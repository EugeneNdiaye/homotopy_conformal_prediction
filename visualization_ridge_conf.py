import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from ridge_conformal_prediction import conf_pred as exact_conf_pred
from approx_conformal_prediction import conf_pred as approx_conf_pred
random_state = 75


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

alpha = 0.1
n_samples, n_features = (100, 50)
X, Y = make_regression(n_samples=n_samples, n_features=n_features,
                       random_state=random_state)
X /= np.linalg.norm(X, axis=0)
Y = (Y - Y.mean()) / Y.std()

lambda_max = np.log(n_features)
n_lambdas = 100
lambdas = lambda_max * np.logspace(0, -4, n_lambdas)


pred_sets = []
length_pred_sets = []

plt.figure()
for lambda_ in lambdas:

    pred_set = exact_conf_pred(X, Y[:-1], lambda_, alpha=0.1)
    plt.vlines(np.log10(lambdas[0] / lambda_),
               pred_set.lower, pred_set.upper, lw=1)
    pred_sets += [pred_set]
    length_pred_sets += [pred_set.upper - pred_set.lower]

plt.hlines(Y[-1], np.log10(lambdas[0] / lambdas[-1]), 0, colors="r",
           label=r"Target $y_{n+1}$")
plt.ylabel("Ridge Conformal Sets")
plt.xlabel(r"$\log_{10}(\lambda_{\max} / \lambda)$")
plt.grid(None)
plt.legend()
plt.tight_layout()
plt.savefig("ridge_confset.pdf", format="pdf")
# plt.show()

# Approximating Ridge conformal set
epsilons = 10. ** (-np.arange(0, 10, 0.5)) * np.linalg.norm(Y[:-1]) ** 2
lambda_ = lambdas[np.argmin(length_pred_sets)]
Y_range = np.min(Y[:-1]), np.max(Y[:-1])

plt.figure()

for epsilon in epsilons:

    pred_set = approx_conf_pred(
        X, Y[:-1], lambda_, Y_range, alpha, epsilon, method="ridge")

    plt.vlines(np.log10(epsilons[0] / epsilon),
               pred_set.lower, pred_set.upper, lw=1)

plt.hlines(Y[-1], -np.log10(epsilon / 5 / epsilons[0]), 0, colors="r",
           label=r"Target $y_{n+1}$")
plt.vlines(np.log10(epsilons[0] / epsilon), pred_set.lower, pred_set.upper,
           lw=1, label=r"$\Gamma^{(\alpha, \epsilon)}$")

# Plot exact conformal set
exact_pred_set = exact_conf_pred(X, Y[:-1], lambda_, alpha=0.1)
plt.vlines(-np.log10(epsilon / 5 / epsilons[0]), pred_set.lower,
           pred_set.upper, lw=2, colors="g", label=r"$\hat \Gamma^{(\alpha)}$")

plt.xlabel(r"$\log_{10}(\epsilon_{\max} / \epsilon)$")
plt.ylabel("Ridge Conformal Sets")
plt.grid(None)
plt.legend()
plt.tight_layout()
plt.savefig("approx_ridge_confset.pdf", format="pdf")
plt.show()
