import numpy as np

class Utilities:
    def error_metrics(actual, predicted):
        actual = actual
        predicted = predicted
        diff = (actual - predicted)
        mse = np.mean(np.square(diff))
        rel = diff / actual
        rmse = np.sqrt(mse)
        bias = 100 * np.median(rel)
        aape = 100 * np.mean(np.abs(rel))
        mape = 100 * np.median(np.abs(rel))
        pe5 = 100 * sum(np.abs(rel) < 0.05) / rel.shape[0]
        pe10 = 100 * sum(np.abs(rel) < 0.10) / rel.shape[0]
        pe20 = 100 * sum(np.abs(rel) < 0.20) / rel.shape[0]
        return [rmse, bias, aape, mape, pe5, pe10, pe20]

