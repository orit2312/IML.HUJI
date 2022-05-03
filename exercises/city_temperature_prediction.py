import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[2]).drop_duplicates()
    positive_vals = ["Year", "Month", "Day", "Temp"]
    for col in positive_vals:
        df = df[df[col] > 0]
    df = df[df["Day"].isin(range(1, 32))]
    df = df[df["Month"].isin(range(1, 13))]
    df = df[df["Year"] < 2023]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:/Users/User/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv")
    y = df.Temp

    # Question 2 - Exploring data for specific country
    df_israel = df[df["Country"] == "Israel"]
    px.scatter(df_israel, x="DayOfYear", y="Temp", color="Year",
               title="Mean Daily Temp as a function of Day of Year").show()

    month_df = df_israel.groupby("Month")
    month_std = month_df.std()

    px.bar(x=month_std.index.to_numpy(), y=month_std.Temp.to_numpy(),
           color_discrete_sequence=(["red", "green", "blue", "goldenrod", "magenta"]),
           title="STD of Daily Temperature per Month",
           labels={"x": "Month", "y": "STD of daily temperature"}).show()

    # Question 3 - Exploring differences between countries
    month_country_df = df.groupby(["Country", "Month"]).agg({"Temp": ["std", "mean"]}).reset_index()
    month_country_df.columns = ["Country", "Month", "STD-Temp", "mean-Temp"]
    px.line(month_country_df, x="Month", y="mean-Temp", color="Country", error_y="STD-Temp",
            title="Average Monthly Temperature with Error").show()

# Question 4 - Fitting model for different values of `k`
    y_israel = df_israel.Temp
    x_israel = df_israel.DayOfYear
    train_x, train_y, test_x, test_y = split_train_test(x_israel, y_israel, 0.75)

    loss_lst = []
    space = range(1, 11)
    for k in space:
        poly = PolynomialFitting(k)
        poly.fit(train_x.to_numpy(), train_y.to_numpy())
        k_loss = np.round(poly.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        print(k_loss)
        loss_lst.append(k_loss)

    px.bar(x=space, y=loss_lst,
           title="Test Error per Polynomial Degree (k)", labels={"x": "polynomial degree", "y": "loss"}).show()

# Question 5 - Evaluating fitted model on different countries
    poly5 = PolynomialFitting(5)
    poly5.fit(x_israel, y_israel)
    losses = dict()
    for country in df["Country"].unique():
        if country != "Israel":
            country_df = df[df["Country"] == country]
            country_loss = poly5.loss(country_df.DayOfYear.to_numpy(), country_df.Temp.to_numpy())
            losses[country] = country_loss

    countries = list(losses.keys())
    loss_lst = list(losses.values())
    px.bar(x=countries, y=loss_lst, title="Polynomial Model of Degree=5 Error over Countries",
           labels={"x": "countries", "y": "losses"}).show()
