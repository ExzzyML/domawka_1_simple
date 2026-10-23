import numpy as np

from .base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    """Линейная регрессия: потеря MSE, обучение спуском или формулой."""

    _solvers = ("gradient", "analytic")

    def fit(self, X, y):
        X, y = self._check_data(X, y)
        self._check_params()
        if self.solver == "analytic":
            self._fit_analytic(X, y)
        else:
            self._fit_gradient(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def _loss(self, X, y):
        error = self.predict(X) - y
        return np.mean(error**2) + self._penalty()

    def _gradient(self, X, y):
        error = self.predict(X) - y
        dw = 2 * X.T @ error / len(y) + self._penalty_gradient()
        db = 2 * np.mean(error)
        return dw, db

    def _fit_analytic(self, X, y):
        if self.penalty not in (None, "l2"):
            raise ValueError(
                f"solver='analytic' работает только с penalty=None или 'l2', "
                f"получено penalty={self.penalty!r}"
            )
        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean()
        else:
            x_mean = np.zeros(X.shape[1])
            y_mean = 0.0
        Xc = X - x_mean
        yc = y - y_mean

        if self.penalty == "l2":
            A = Xc.T @ Xc + len(y) * self.alpha * np.eye(X.shape[1])
            self.coef_ = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
        else:
            self.coef_ = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.intercept_ = y_mean - x_mean @ self.coef_

        self.loss_history_ = [self._loss(X, y)]
        self.n_iter_ = 1
