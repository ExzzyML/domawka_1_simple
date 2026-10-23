import numpy as np

from .base import BaseLinearModel


class LogisticRegression(BaseLinearModel):
    """Логистическая регрессия: два класса, потеря log loss, градиентный спуск."""

    def __init__(
        self,
        fit_intercept=True,
        solver="gradient",
        penalty=None,
        alpha=0.0,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        verbose=False,
        random_state=None,
        threshold=0.5,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            solver=solver,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )
        self.threshold = threshold

    def fit(self, X, y):
        X, y = self._check_data(X, y)
        self._check_params()
        self._check_target(y)
        self._fit_gradient(X, y)
        return self

    def predict_proba(self, X):
        p = self._proba(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._proba(X) >= self.threshold).astype(int)

    def _loss(self, X, y):
        p = np.clip(self._proba(X), 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return log_loss + self._penalty()

    def _gradient(self, X, y):
        error = self._proba(X) - y
        dw = X.T @ error / len(y) + self._penalty_gradient()
        db = np.mean(error)
        return dw, db

    def _check_params(self):
        super()._check_params()
        if not 0 < self.threshold < 1:
            raise ValueError(
                f"threshold должен быть между 0 и 1, получено {self.threshold}"
            )

    @staticmethod
    def _check_target(y):
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y для логистической регрессии должен состоять из 0 и 1")

    def _proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
