from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()

    irrelevant_features = ["id", "date", "lat", "long"]
    for feature in irrelevant_features:
        df = df.drop(feature, axis='columns')

    positive_features = ["price", "sqft_living", "sqft_lot", "grade", "yr_built", "sqft_living15", "sqft_lot15"]

    for feature in positive_features:
        df = df[df[feature] > 0]

    df = pd.get_dummies(df, prefix=["yr_renovated_", "zipcode_"],
                        columns=["yr_renovated", "zipcode"])
    df = df[df["condition"].isin(range(1, 6))]
    df["yr_built"] = (df["yr_built"] / 10).astype(int) - 190

    prices = df["price"]
    final_df = df.drop("price", axis='columns')
    return final_df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    cols_to_remove = ["yr_renovated_", "zipcode_"]
    for col in cols_to_remove:
        col_to_remove = [c for c in X.columns if c.lower()[:len(col)] != col]
        X = X[col_to_remove]

    y_std = np.std(y)
    for feature in X:
        feature_std = np.std(X[feature])
        cov = np.cov(X[feature], y)[0, 1]
        pearson_correlation = cov / (y_std * feature_std)
        fig = go.Figure([go.Scatter(x=X[feature], y=y, mode='markers')], layout=go.Layout(
            title=f"Pearson Correlation between {feature} Values and the Prices is {pearson_correlation}",
            xaxis=dict(title=f"{feature}"), yaxis=dict(title="Prices"), height=300, width=1000))
        fig.write_image(output_path + f"{feature} person correlation.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("C:/Users/User/PycharmProjects/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:/Users/User/PycharmProjects/IML.HUJI/plots")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, .75)
    training_set = train_x, train_y
    test_set = test_x, test_y

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss = []
    std = []
    lr_obj = LinearRegression(True)
    for p in range(10, 101):
        p_losses = []
        for counter in range(10):
            frac = p / 100
            new_train_x, new_train_y, a, b = split_train_test(train_x, train_y, frac)
            lr_obj.fit(new_train_x.to_numpy(), new_train_y.to_numpy())
            loss = lr_obj.loss(test_x.to_numpy(), test_y.to_numpy())
            p_losses.append(loss)
        mean_loss.append(np.mean(p_losses))
        std.append(np.std(p_losses))

    p_vals = np.linspace(10, 100, 91)
    mean_loss, std = np.array(mean_loss), np.array(std)
    go.Figure([go.Scatter(x=p_vals, y=mean_loss, mode="lines", name="Mean Prediction",
                    marker=dict(color="black")),
                    go.Scatter(x=p_vals, y=(mean_loss - 2 * std), mode="lines",
                    marker=dict(color="red")),
                    go.Scatter(x=p_vals, y=(mean_loss + 2 * std), mode="lines", marker=dict(color="red"))],
                    layout=go.Layout(
                        title="Mean Loss as Function of Percentage with Noise",
                        xaxis=dict(title="percentages values"), yaxis=dict(title="mean loss"))).show()


