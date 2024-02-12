from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Trainer:
    """Trainer Module."""

    def __init__(
        self,
        datamodule: Callable,
        model: Callable,
    ):
        """Initializes the Trainer object.

        Parameters:
            datamodule (Callable): A callable object that provides the data module.
            model (Callable): A callable object that represents the model.
            random_state (int): The random state for reproducibility. Default is 42.
        """
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = datamodule.split()

    def fit_cv(
        self,
        model_name: str,
        search_strategy: Callable,
        param_distributions: dict,
        logger: Callable,
    ):
        """Fits the model using cross-validation and optimizes hyperparameters.

        Parameters:
            search_strategy (Callable): The search strategy for hyperparameter optimization.
            param_distributions (dict): Dictionary of hyperparameter distributions.

        Returns:
            tuple: A tuple containing the optimized scores, test scores, and best hyperparameters.
        """
        logger.info(f"Training {model_name}...")
        if model_name == "LogisticRegression":
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", self.model()),
                ]
            )
            param_distributions = {f"model__{k}": v for k, v in param_distributions.items()}
        else:
            model = self.model
        search_strategy = search_strategy(estimator=model, param_distributions=param_distributions)

        logger.info("Optimizing hyperparameters...")
        search_strategy.fit(self.X_train, self.y_train)

        logger.info("Retrieving hparams and results...")
        hparams = search_strategy.best_params_
        opt_score = {
            "hparam_opt/train_cv/acc": search_strategy.cv_results_["mean_train_score"][
                search_strategy.best_index_
            ],
            "hparam_opt/test_cv/acc": search_strategy.cv_results_["mean_test_score"][
                search_strategy.best_index_
            ],
        }

        test_score = {"test/acc": search_strategy.score(self.X_test, self.y_test)}

        return opt_score, test_score, hparams

    @staticmethod
    def fit(
        model: Callable,
        model_name: str,
        X_train: np.array,
        y_train: np.array,
        hparams: dict,
    ):
        """Fits a given model to the training data.

        Parameters:
            model (Callable): The model to be trained.
            X_train (np.array): The input features for training.
            y_train (np.array): The target labels for training.
            hparams (dict): Hyperparameters for the model.

        Returns:
            The trained model.
        """
        if model_name == "LogisticRegression":
            hparams = {k.split("__")[1]: v for k, v in hparams.items()}

            model = Pipeline(steps=[("scaler", StandardScaler()), ("model", model(**hparams))])
        else:
            model = model(**hparams)

        model.fit(X_train, y_train)
        return model

    @staticmethod
    def predict(model: Callable, X_test: np.array):
        """Predicts the output labels for the given test data using the specified model.

        Parameters:
            model (Callable): The trained model used for prediction.
            X_test (np.array): The test data.

        Returns:
            np.array: The predicted output labels.
        """
        y_test = model.predict(X_test)
        return y_test

    @staticmethod
    def evaluate(y_test: np.array, y_pred: np.array):
        """Evaluate the performance of the model by calculating the accuracy score.

        Parameters:
            y_test (np.array): The true labels of the test data.
            y_pred (np.array): The predicted labels of the test data.

        Returns:
            dict: A dictionary containing the accuracy score.
        """
        score = {
            "test/acc": accuracy_score(y_test, y_pred),
        }
        return score
