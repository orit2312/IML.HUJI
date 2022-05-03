from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import scipy.stats


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, n_k = np.unique(y, return_counts=True)
        mu_mat = []
        n_samples, n_features = X.shape

        for j in range(len(self.classes_)):
            k_mu = np.zeros(n_features)
            for i, yi in enumerate(y):
                if yi == self.classes_[j]:     #classes[j] = k
                    k_mu += X[i]
            k_mu = k_mu / n_k[j]
            mu_mat.append(k_mu)
        self.mu_ = np.array(mu_mat)

        self.cov_ = np.ndarray((n_features, n_features))
        for i, k in enumerate(self.classes_):
            x_sub_mu = X[y == k] - self.mu_[i]
            self.cov_ += x_sub_mu.T @ x_sub_mu
        self.cov_ = (1 / (n_samples - len(self.classes_))) * self.cov_

        self._cov_inv = inv(self.cov_)

        self.pi_ = n_k / n_samples

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_hat = []
        for sam_ind in range(X.shape[0]):
            vals = []
            for k_ind in range(len(self.classes_)):
                mu_k = np.transpose(self.mu_[k_ind])
                a_k = self._cov_inv @  mu_k
                b_k = np.log(self.pi_[k_ind]) - 0.5 * self.mu_[k_ind] @ a_k
                val_k = a_k @ X[sam_ind] + b_k
                vals.append(val_k)
            res = np.argmax(vals)
            y_hat.append(res)
        return np.array(y_hat)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples = X.shape[0]
        likelihood_mat = np.zeros((n_samples, len(self.classes_)))
        for k in range(len(self.classes_)):
            prob_per_k = scipy.stats.multivariate_normal(self.mu_[k], self.cov_).logpdf(X) + np.log(self.pi_[k])
            likelihood_mat[:, k] = prob_per_k
        return likelihood_mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)


