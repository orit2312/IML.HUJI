from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    uni_random = UnivariateGaussian()  # false not needded- defalt?
    uni_random.fit(X)
    print((uni_random.mu_, uni_random.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mu = 10
    mu_dists = []
    space = np.linspace(10, 1000, 100).astype(np.int)
    for sam in space:
        sample_mu = np.mean(X[:sam])
        mu_dist = np.abs(sample_mu - mu)
        mu_dists.append(mu_dist)

    go.Fifure([go.Scatter(x=space, y=mu_dist, mode='markers+lines', name=r'$\widehat\mu$')],
              layout = go.Layout(title=r"$\text{(2) The Distance Between Estimated And True Value Of The Expectation As Functiom Of Samples Number}$",
        xaxis_title="$m\\text{number of samples}$", yaxis_title="r$\{distance}\mu$", height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uni_random.pdf(X)

    go.Fifure([go.Scatter(x=X, y=pdf, mode='markers+lines', name=r'$\widehat\mu$')],
               layout = go.Layout(title=r"$\text{(3) PDF}$", xaxis_title="$m\\text{samples}$",
                                  yaxis_title="r$\{pdf(x)}\mu$", height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_random = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    size = 1000
    x = np.random.multivariate_normal(mu, cov, size)
    multi_random.fit(x)
    print(multi_random.mu_)
    print(multi_random.cov_)

    # Question 5 - Likelihood evaluation
    space = np.linspace(-10, 10, 200)
    log_likelihood_mat = []

    max_val = np.NINF
    f1_max, _f3_max = -10, -10

    for f1 in space:
        row_vals = []
        for f3 in space:
            mu = np.array([f1, 0, f3, 0])
            log_likelihood = multi_random.log_likelihood(mu, cov, x)
            row_vals.append(log_likelihood)

            if max_val < log_likelihood:  # for Question 6
                max_val = log_likelihood
                f1_max, f3_max = f1, f3


    log_likelihood_mat.append(row_vals)

    go.Figure(data=go.Heatmap(x=space, y=space, z=log_likelihood_mat),
          layout=go.Layout(title=r"$\text{Log-Likelihood Heatmap}$"),
          xaxis=dict(title=r"$\text{f1 values}$"),
          yaxis=dict(title=r"$\text{f3 values}$")).show()


# Question 6 - Maximum likelihood
    f1_max, f3_max = round(f1, 3), round(f3, 3)
    print(f1_max, f3_max)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
