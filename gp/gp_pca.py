import numpy as np
import torch
import gpytorch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_ITERS = 200
LR = 0.05
PCA_DIM = 20  # components kept from the whole feature matrix


class ExactGP(gpytorch.models.ExactGP):
    """single ARD RBF kernel over the PCA components"""

    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        f = X.shape[1]
        self.mean = gpytorch.means.ConstantMean()
        self.covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=f,
                lengthscale_constraint=gpytorch.constraints.Interval(1e-2, 1e2),
            ),
            outputscale_constraint=gpytorch.constraints.Interval(1e-3, 1e3),
        )

    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(self.mean(X), self.covar(X))


def reduce(X_train, X_test, blocks=None, n_components=PCA_DIM):
    """one PCA over all features, fit on the train split only.
    If blocks are given, each block is first scaled to unit total variance so a wide or
    high-variance block (e.g. a 1024-d embedding) does not dominate the components."""
    mu = X_train.mean(axis=0)
    X_train, X_test = X_train - mu, X_test - mu
    if blocks is not None:
        start = 0
        for width in blocks:
            cols = slice(start, start + width)
            scale = np.sqrt(X_train[:, cols].var(axis=0).sum()) or 1.0  # constant block -> leave as is
            X_train[:, cols] /= scale
            X_test[:, cols] /= scale
            start += width
    k = min(n_components, X_train.shape[1], len(X_train) - 1)
    if k < X_train.shape[1]:
        pca = PCA(n_components=k).fit(X_train)
        X_train, X_test = pca.transform(X_train), pca.transform(X_test)
    # unit variance per component so the lengthscale bounds mean the same thing for every component
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return X_train / sd, X_test / sd


def fit_predict(X_train, y_train, X_test):
    """fits an ARD RBF GP on the train split and returns the predictive mean on the test split"""
    # standardize y like sklearn's normalize_y=True
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(1e-6, 1e1)
    ).to(DEVICE)
    model = ExactGP(X_train, y_train, likelihood).to(DEVICE)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(TRAIN_ITERS):
        optimizer.zero_grad()
        loss = -mll(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X_test)).mean
    return pred * y_std + y_mean


def e(F, blocks=None, n_splits=5, n_components=PCA_DIM):
    """F is data points x [y, *features]; blocks are the feature block widths, used only to
    balance blocks before the PCA. Returns mean absolute error averaged over the CV folds."""
    F = np.asarray(F, dtype=float)
    y = F[:, 0]
    X = F[:, 1:]
    if blocks is not None:
        assert sum(blocks) == X.shape[1], f"blocks sum to {sum(blocks)} but F has {X.shape[1]} features"

    keep = ~np.isnan(y)  # drop rows with no PL value
    X, y = X[keep], y[keep]
    if len(y) < n_splits:
        raise ValueError(f"only {len(y)} data points with a PL value, need at least {n_splits}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    errors = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = reduce(X[train_idx], X[test_idx], blocks, n_components)
        X_train = torch.as_tensor(X_train, dtype=torch.float64, device=DEVICE)
        X_test = torch.as_tensor(X_test, dtype=torch.float64, device=DEVICE)
        y_train = torch.as_tensor(y[train_idx], dtype=torch.float64, device=DEVICE)
        y_test = torch.as_tensor(y[test_idx], dtype=torch.float64, device=DEVICE)
        pred = fit_predict(X_train, y_train, X_test)
        errors.append((pred - y_test).abs().mean().item())
    return float(np.mean(errors))
