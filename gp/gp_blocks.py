# Additive GP: one ARD RBF kernel per feature block, after per-block standardize + PCA (fit per fold).
import numpy as np
import torch
import gpytorch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_ITERS = 200
LR = 0.05
PCA_DIM = 10  # max components kept per feature block


class ExactGP(gpytorch.models.ExactGP):
    """sum of one ARD RBF kernel per feature block, so each block gets its own output scale"""

    def __init__(self, X, y, likelihood, blocks):
        super().__init__(X, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        kernels = []
        start = 0
        for width in blocks:
            kernels.append(gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=width,
                    active_dims=torch.arange(start, start + width),
                    lengthscale_constraint=gpytorch.constraints.Interval(1e-2, 1e2),
                ),
                outputscale_constraint=gpytorch.constraints.Interval(1e-3, 1e3),
            ))
            start += width
        self.covar = gpytorch.kernels.AdditiveKernel(*kernels)

    def forward(self, X):
        return gpytorch.distributions.MultivariateNormal(self.mean(X), self.covar(X))


def reduce_blocks(X_train, X_test, blocks, n_components=PCA_DIM):
    """standardizes each block and projects it onto at most n_components PCs, fit on the train split only"""
    train, test, widths = [], [], []
    start = 0
    for width in blocks:
        tr, te = X_train[:, start:start + width], X_test[:, start:start + width]
        start += width
        scaler = StandardScaler().fit(tr)
        tr, te = scaler.transform(tr), scaler.transform(te)
        k = min(n_components, width, len(tr) - 1)
        if k < width:
            pca = PCA(n_components=k).fit(tr)
            tr, te = pca.transform(tr), pca.transform(te)
        train.append(tr)
        test.append(te)
        widths.append(tr.shape[1])
    return np.hstack(train), np.hstack(test), widths


def fit_predict(X_train, y_train, X_test, blocks):
    """fits the additive block GP on the train split and returns the predictive mean on the test split"""
    # standardize y like sklearn's normalize_y=True
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(1e-6, 1e1)
    ).to(DEVICE)
    model = ExactGP(X_train, y_train, likelihood, blocks).to(DEVICE)

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
    """F is data points x [y, *features]; blocks are the feature block widths (default: one block).
    Returns mean absolute error averaged over the CV folds."""
    F = np.asarray(F, dtype=float)
    y = F[:, 0]
    X = F[:, 1:]
    blocks = list(blocks) if blocks is not None else [X.shape[1]]
    assert sum(blocks) == X.shape[1], f"blocks sum to {sum(blocks)} but F has {X.shape[1]} features"

    keep = ~np.isnan(y)  # drop rows with no PL value
    X, y = X[keep], y[keep]
    if len(y) < n_splits:
        raise ValueError(f"only {len(y)} data points with a PL value, need at least {n_splits}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    errors = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test, widths = reduce_blocks(X[train_idx], X[test_idx], blocks, n_components)
        X_train = torch.as_tensor(X_train, dtype=torch.float64, device=DEVICE)
        X_test = torch.as_tensor(X_test, dtype=torch.float64, device=DEVICE)
        y_train = torch.as_tensor(y[train_idx], dtype=torch.float64, device=DEVICE)
        y_test = torch.as_tensor(y[test_idx], dtype=torch.float64, device=DEVICE)
        pred = fit_predict(X_train, y_train, X_test, widths)
        errors.append((pred - y_test).abs().mean().item())
    return float(np.mean(errors))
