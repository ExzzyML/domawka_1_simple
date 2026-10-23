import numpy as np

from linear_models import LogisticRegression

# Простые данные: маленькие x → класс 0, большие → класс 1
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([0, 0, 1, 1])


def test_create_model():
    model = LogisticRegression(penalty="l2", alpha=0.1)
    assert model.penalty == "l2"
    assert model.alpha == 0.1
    assert model.threshold == 0.5  # порог по умолчанию


def test_fit_returns_self():
    model = LogisticRegression()
    assert model.fit(X, y) is model


def test_predict():
    model = LogisticRegression(learning_rate=0.1).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (4,)
    assert np.array_equal(pred, y)  # простые данные модель угадывает полностью


def test_predict_proba():
    model = LogisticRegression(learning_rate=0.1).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1)  # P(0) + P(1) = 1 в каждой строке


def test_coef_and_intercept_after_fit():
    model = LogisticRegression().fit(X, y)
    assert model.coef_.shape == (1,)
    assert hasattr(model, "intercept_")


def test_fit_intercept_false():
    model = LogisticRegression(fit_intercept=False).fit(X, y)
    assert model.intercept_ == 0.0


def test_l1_and_l2_shrink_weight():
    plain = LogisticRegression(learning_rate=0.1).fit(X, y)
    l1 = LogisticRegression(learning_rate=0.1, penalty="l1", alpha=0.1).fit(X, y)
    l2 = LogisticRegression(learning_rate=0.1, penalty="l2", alpha=0.1).fit(X, y)
    assert abs(l1.coef_[0]) < abs(plain.coef_[0])
    assert abs(l2.coef_[0]) < abs(plain.coef_[0])


def test_loss_history():
    model = LogisticRegression(learning_rate=0.1).fit(X, y)
    assert len(model.loss_history_) == model.n_iter_
    assert model.loss_history_[-1] < model.loss_history_[0]
