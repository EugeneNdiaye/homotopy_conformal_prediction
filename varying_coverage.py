import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from ridge_conformal_prediction import conf_pred as ridge_conf_pred
import intervals
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
lambda_ = lambdas[50]

alphas = np.arange(1, 9) / 10
lengths = np.empty(alphas.shape[0])
for i_alpha, alpha in enumerate(alphas):

    ridge_set = ridge_conf_pred(X, Y[:-1], lambda_, alpha=alpha)
    # taking the convex hull
    ridge_set = intervals.closed(ridge_set.lower, ridge_set.upper)
    lengths[i_alpha] = ridge_set.upper - ridge_set.lower


plt.figure()
plt.plot(alphas, lengths)
plt.xlabel("Coverage" + r"$\alpha$")
plt.ylabel("Length of " + r"$\Gamma(x_{n+1})$")
plt.tight_layout()
plt.show()
