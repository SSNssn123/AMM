import sklearn.metrics as met
import numpy as np


def compute_rpd(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    std_dev = np.std(y_true)
    rpd = std_dev / rmse
    return rpd

def get_regression_metrics(y_test, y_pred):
    return (
        met.r2_score(y_test, y_pred),
        np.sqrt(met.mean_squared_error(y_test, y_pred)),
        compute_rpd(y_test, y_pred),
        met.mean_absolute_error(y_test, y_pred),)