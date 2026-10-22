import numpy as np

from .base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    """Линейная регрессия: потеря MSE, обучение градиентным спуском."""

    def fit(self, X, y):
        X, y = self._check_data(X, y)
        self._check_params()
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
