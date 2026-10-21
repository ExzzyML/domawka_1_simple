from abc import ABC, abstractmethod


class BaseLinearModel(ABC):
    """Общий родитель линейных моделей: настройки, проверки, обучение спуском."""

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

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def _loss(self, X, y): ...

    @abstractmethod
    def _gradient(self, X, y): ...
