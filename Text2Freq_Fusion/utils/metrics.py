import numpy as np
from fastdtw import fastdtw  
from scipy.spatial.distance import euclidean

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def DTW(pred, true):
    pred = pred.squeeze(2)
    true = true.squeeze(2)
    # Compute DTW Distance using fastdtw
    batch_size = pred.shape[0]
    dtw_score = 0.0
    for i in range(batch_size):
        distance, _ = fastdtw(pred[i], true[i], dist=2)
        dtw_score += distance
    dtw_score = dtw_score / batch_size
    
    return dtw_score

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    dtw = DTW(pred, true)
    return mae, mse, rmse, mape, mspe, dtw
