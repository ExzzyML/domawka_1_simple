import numpy as np

from linear_models import LinearRegression

# Простые данные: y = 2x + 1, правильный ответ coef_ = 2, intercept_ = 1
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])


def test_create_model():
    model = LinearRegression(solver="analytic", penalty="l2", alpha=0.5)
    assert model.solver == "analytic"
    assert model.penalty == "l2"
    assert model.alpha == 0.5


def test_fit_returns_self():
    model = LinearRegression()
    assert model.fit(X, y) is model


def test_predict_shape():
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (4,)


def test_coef_and_intercept_after_fit():
    model = LinearRegression()
    assert not hasattr(model, "coef_")  # до fit атрибутов нет
    model.fit(X, y)
    assert model.coef_.shape == (1,)  # по одному весу на признак
    assert hasattr(model, "intercept_")


def test_analytic_solver():
    model = LinearRegression(solver="analytic").fit(X, y)
    assert np.isclose(model.coef_[0], 2)
    assert np.isclose(model.intercept_, 1)


def test_gradient_solver():
    model = LinearRegression(solver="gradient", learning_rate=0.1, tol=1e-10)
    model.fit(X, y)
    assert np.isclose(model.coef_[0], 2, atol=0.01)
    assert np.isclose(model.intercept_, 1, atol=0.01)


def test_fit_intercept_false():
    model = LinearRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0


def test_no_penalty():
    # без штрафа alpha ни на что не влияет: ответ тот же, что без alpha
    model = LinearRegression(solver="analytic", penalty=None, alpha=10.0).fit(X, y)
    assert np.isclose(model.coef_[0], 2)


def test_l2_shrinks_weight():
    model = LinearRegression(penalty="l2", alpha=1.0).fit(X, y)
    assert 0 < model.coef_[0] < 2  # штраф уменьшил вес


def test_l1_shrinks_weight():
    model = LinearRegression(penalty="l1", alpha=1.0).fit(X, y)
    assert 0 < model.coef_[0] < 2  # штраф уменьшил вес


def test_early_stopping():
    model = LinearRegression(max_iter=10_000).fit(X, y)
    assert model.n_iter_ < 10_000  # остановились раньше потолка


def test_n_iter():
    model = LinearRegression(max_iter=50, tol=0).fit(X, y)  # tol = 0 → без остановки
    assert model.n_iter_ == 50


def test_loss_history():
    model = LinearRegression().fit(X, y)
    assert len(model.loss_history_) == model.n_iter_  # одна запись на итерацию
    assert model.loss_history_[-1] < model.loss_history_[0]  # потеря уменьшилась
