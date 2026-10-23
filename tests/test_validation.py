import numpy as np
import pytest

from linear_models import LinearRegression, LogisticRegression

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([0, 0, 1, 1])


def test_invalid_solver():
    with pytest.raises(ValueError):
        LinearRegression(solver="banana").fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression(solver="banana").fit(X, y)


def test_invalid_penalty():
    with pytest.raises(ValueError):
        LinearRegression(penalty="l3").fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression(penalty="l3").fit(X, y)


def test_invalid_training_params():
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=0).fit(X, y)
    with pytest.raises(ValueError):
        LinearRegression(max_iter=0).fit(X, y)
    with pytest.raises(ValueError):
        LinearRegression(tol=-1).fit(X, y)
    with pytest.raises(ValueError):
        LinearRegression(alpha=-1).fit(X, y)


def test_x_and_y_different_size():
    with pytest.raises(ValueError):
        LinearRegression().fit(X, y[:3])  # в X 4 строки, в y 3 числа


def test_logistic_invalid_y():
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.array([0, 1, 2, 1]))  # класса 2 не бывает


def test_unsupported_solver_penalty():
    # для L1 нет формулы, поэтому analytic + l1 нельзя
    with pytest.raises(ValueError):
        LinearRegression(solver="analytic", penalty="l1").fit(X, y)
    # у логистической регрессии есть только спуск
    with pytest.raises(ValueError):
        LogisticRegression(solver="analytic").fit(X, y)
