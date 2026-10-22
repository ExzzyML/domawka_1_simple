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

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def _loss(self, X, y): ...

    @abstractmethod
    def _gradient(self, X, y): ...
