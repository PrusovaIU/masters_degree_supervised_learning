from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linear_regression import LinearRegression


def show_predictions(
        x: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        png_path: Path = Path("output.png")
) -> None:
    """
    Отображение результатов предсказания.

    :param x: Признаки.
    :param y: Целевые переменные.
    :param y_pred: Предсказанные значения.
    :param png_path: Путь для сохранения изображения.
    :return: None.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="red", label="Фактические значения")
    plt.plot(
        x,
        y_pred,
        color="blue",
        linewidth=2,
        label="Прогнозируемые значения"
    )
    plt.title("График линейной регрессии")
    plt.xlabel("Количество лет опыта")
    plt.ylabel("Зарплата")
    plt.legend()
    plt.savefig(png_path)
    print(f"Результаты сохранены в {png_path.absolute()}")


def lr_model(file_path: str) -> None:
    """
    Модель линейной регрессии.

    :param file_path: Путь к файлу с данными.
    :return: None.
    """
    df = pd.read_csv(file_path)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, 1].values

    model = LinearRegression(iterations=1000, learning_rate=0.01)
    model.fit(x, y)

    y_pred = model.predict(x)

    print("Weights: ", *np.round(model.weights, 2))
    print("Bias: ", np.round(model.bias, 2))
    show_predictions(x, y, y_pred)


if __name__ == "__main__":
    parser = ArgumentParser(description="Linear Regression")
    parser.add_argument(
        "--file",
        default="salary_data.csv",
        type=str,
        help="Path to the dataset file"
    )
    args = parser.parse_args()
    lr_model(args.file)
