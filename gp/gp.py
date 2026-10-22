import numpy as np
import torch
import gpytorch
from sklearn.model_selection import KFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_ITERS = 200
LR = 0.05


class ExactGP(gpytorch.models.ExactGP):
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


def e(F, blocks=None, n_splits=5, seed=0):
    """F is data points x [y, *features]; returns mean absolute error averaged over the CV folds.
    blocks is accepted for the same call signature as gp_blocks / gp_pca and is ignored here."""
    F = np.asarray(F, dtype=float)
    y = F[:, 0]
    X = F[:, 1:]

    keep = ~np.isnan(y)  # drop rows with no PL value
    X, y = X[keep], y[keep]
    if len(y) < n_splits:
        raise ValueError(f"only {len(y)} data points with a PL value, need at least {n_splits}")

    X = torch.as_tensor(X, dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(y, dtype=torch.float64, device=DEVICE)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errors = []
    for train_idx, test_idx in cv.split(X.cpu().numpy()):
        pred = fit_predict(X[train_idx], y[train_idx], X[test_idx])
        errors.append((pred - y[test_idx]).abs().mean().item())
    return float(np.mean(errors))
