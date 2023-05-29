import numpy as np
import warnings
from scipy.stats import spearmanr


# MEASURES OF PERFORMANCE OF THE MODEL

def get_spearman_btw_timeseries(real_ts, pred_ts):
    if real_ts.shape != pred_ts.shape:
        warnings.warn("Shapes of the input time series are different!", Warning)
    spearman_list = []
    for i in range(len(real_ts)):
        spearman_list.append(spearmanr(real_ts[i], pred_ts[i])[0])
    return spearman_list