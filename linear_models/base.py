from abc import ABC, abstractmethod

import numpy as np


class BaseLinearModel(ABC):
    """Общий родитель линейных моделей: настройки, проверки, обучение спуском."""

    _solvers = ("gradient",)
    _penalties = (None, "l1", "l2")

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
    ):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _check_params(self):
        if self.solver not in self._solvers:
            raise ValueError(
                f"solver={self.solver!r} не поддерживается, доступны: {self._solvers}"
            )
        if self.penalty not in self._penalties:
            raise ValueError(
                f"penalty={self.penalty!r} не поддерживается, доступны: {self._penalties}"
            )
        if not self.learning_rate > 0:
            raise ValueError(
                f"learning_rate должен быть больше 0, получено {self.learning_rate}"
            )
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError(
                f"max_iter должен быть целым числом не меньше 1, получено {self.max_iter}"
            )
        if not self.tol >= 0:
            raise ValueError(f"tol не может быть отрицательным, получено {self.tol}")
        if not self.alpha >= 0:
            raise ValueError(
                f"alpha не может быть отрицательным, получено {self.alpha}"
            )

    @staticmethod
    def _check_data(X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X должен быть двумерным массивом, получено ndim={X.ndim}"
            )
        if y.ndim != 1:
            raise ValueError(
                f"y должен быть одномерным массивом, получено ndim={y.ndim}"
            )
        if len(X) != len(y):
            raise ValueError(f"разное число объектов: в X {len(X)}, в y {len(y)}")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("в X или y есть NaN или бесконечность")
        return X, y

    def _penalty(self):
        w = self.coef_
        if self.penalty == "l2":
            return self.alpha * np.sum(w**2)
        if self.penalty == "l1":
            return self.alpha * np.sum(np.abs(w))
        return 0.0

    def _penalty_gradient(self):
        w = self.coef_
        if self.penalty == "l2":
            return 2 * self.alpha * w
        if self.penalty == "l1":
            return self.alpha * np.sign(w)
        return np.zeros_like(w)

    def _fit_gradient(self, X, y):
        """Общий градиентный спуск: веса шаг за шагом двигаются против наклона."""
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []

        for i in range(self.max_iter):
            dw, db = self._gradient(X, y)
            self.coef_ = self.coef_ - self.learning_rate * dw
            if self.fit_intercept:
                self.intercept_ = self.intercept_ - self.learning_rate * db

            loss = self._loss(X, y)
            if not np.isfinite(loss):
                raise ValueError(
                    f"обучение разошлось на итерации {i}: уменьшите learning_rate"
                )
            self.loss_history_.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"итерация {i}: потеря {loss:.6f}")

            if i > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                break

        self.n_iter_ = len(self.loss_history_)

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def _loss(self, X, y): ...

    @abstractmethod
    def _gradient(self, X, y): ...
