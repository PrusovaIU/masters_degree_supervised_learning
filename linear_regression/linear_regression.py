import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float, iterations: int):
        """
        Конструктор класса LinearRegression.

        :param learning_rate: Скорость обучения (шаг градиентного спуска).
        :param iterations: Количество итераций для оптимизации.
        """
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._weights = None  # Веса модели
        self._bias = None  # Смещение модели

    @property
    def weights(self) -> np.ndarray | None:
        """
        :return: Веса модели, если модель обучена, иначе None.
        """
        return self._weights

    @property
    def bias(self) -> float | None:
        """
        :return: Смещение модели, если модель обучена, иначе None.
        """
        return self._bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Метод для обучения модели с использованием градиентного спуска.

        :param x: Матрица признаков.
        :param y: Вектор истинных меток.

        :return: None
        """
        n_samples, n_features = x.shape

        self._weights = np.zeros(n_features)
        self._bias = 0

        for _ in range(self._iterations):
            self.update_weights(x, y)

    def update_weights(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Вспомогательный метод для обновления весов и смещения.

        :param x: Матрица признаков.
        :param y: Вектор истинных меток.

        :return: None.
        """
        y_pred = np.dot(x, self._weights) + self._bias

        # Вычисление производных dL/dw и dL/db
        dw = -(2 / len(y)) * sum(
            (y[i] - y_pred[i]) * x[i][0] for i in range(len(y))
        )
        db = -(2 / len(y)) * sum(
            y[i] - y_pred[i] for i in range(len(y))
        )

        self._weights -= self._learning_rate * dw
        self._bias -= self._learning_rate * db

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Предсказываем значения на новых данных.

        :param x: Матрица признаков.

        :return: Массив предсказанных значений.
        """
        return np.dot(x, self._weights) + self._bias
