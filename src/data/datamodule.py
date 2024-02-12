from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataModule:
    """Datamodule class for preparing and splitting the data."""

    def __init__(
        self,
        filepath: str,
        label_col: str,
        feature_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the DataModule object.

        Parameters:
            dataset (pd.DataFrame): The input dataset.
            label_col (str): The name of the column containing the labels.
            feature_cols (List[str]): The list of column names containing the features.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The random seed for reproducible results. Defaults to 42.
        """
        self.dataset = pd.read_parquet(filepath)
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self):
        """Prepare the data by extracting the features and labels from the dataset.

        Returns:
            features (numpy.ndarray): Array of features.
            labels (numpy.ndarray): Array of labels.
        """
        features = self.dataset[self.feature_cols].values
        labels = self.dataset[self.label_col].values
        return features, labels

    def encode_labels(self, labels: np.array, problem_type: str = "multi-class") -> np.array:
        """Encodes the labels based on the specified problem type.

        Parameters:
            labels (np.array): The array of labels to be encoded.
            problem_type (str): The type of classification problem. Default is "multi-class".

        Returns:
            np.array: The encoded labels.
        """
        if problem_type == "multi-class":
            enc = LabelEncoder()
            labels = enc.fit_transform(labels)
        elif problem_type == "multi-label":
            raise NotImplementedError("Multi-label classification is not yet supported")
        else:
            raise ValueError(
                "Invalid problem type. Please choose either 'multi-class' or 'multi-label'"
            )
        return labels

    def split(self):
        """Splits the data into training and testing sets.

        Returns:
            X_train (np.array): Training features.
            X_test (np.array): Testing features.
            y_train (np.array): Training labels.
            y_test (np.array): Testing labels.
        """
        features, labels = self.prepare_data()
        labels = self.encode_labels(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        return X_train, X_test, y_train, y_test
