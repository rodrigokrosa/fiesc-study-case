import numpy as np
import pandas as pd


def remove_nans(input_array: np.array) -> np.array:
    """Remove NaN values from the input array.

    Parameters:
        input_array (np.array): The input array containing NaN values.

    Returns:
        np.array: The input array with NaN values removed.
    """
    return input_array[~np.isnan(input_array)]


def load_raw_data():
    """Load raw data from files and create a dataset.

    Returns:
        dataset (pd.DataFrame): The dataset containing the loaded raw data.
    """
    classes = np.load("data/raw/Classes.npy", allow_pickle=True)
    df_classes = pd.DataFrame(classes, columns=["classes"])

    sensor_1 = np.load("data/raw/Dados_1.npy", allow_pickle=True)
    sensor_2 = np.load("data/raw/Dados_2.npy", allow_pickle=True)
    sensor_3 = np.load("data/raw/Dados_3.npy", allow_pickle=True)
    sensor_5 = np.load("data/raw/Dados_5.npy", allow_pickle=True)

    sensor_1, sensor_2, sensor_3 = sensor_1[:, :-1], sensor_2[:, :-1], sensor_3[:, :-1]

    sensor_data = pd.DataFrame(
        {
            "sensor_1/acceleration": list(sensor_1),
            "sensor_2/acceleration": list(sensor_2),
            "sensor_3/acceleration": list(sensor_3),
            "sensor_5/acceleration": list(sensor_5),
        }
    )
    dataset = pd.concat([df_classes, sensor_data], axis=1)

    dataset["sensor_5/acceleration"] = dataset["sensor_5/acceleration"].apply(
        lambda x: remove_nans(x)
    )

    return dataset
