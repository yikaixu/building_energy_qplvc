import numpy as np
import math
from tqdm import tqdm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.interpolate import BSpline


def qnis(y, x, t, tau, width, k, dn):
    # get conditional sample quantile of y
    t_min = np.min(t) - 0.05
    t_max = np.max(t) + 0.05
    t_region_nums = math.ceil((t_max - t_min) / (2 * width))
    t_regions = [(id * 2 * width + t_min, (id + 1) * 2 * width + t_min) if (id + 1) * 2 * width + t_min <= t_max else (id * 2 * width + t_min, t_max) for id in range(t_region_nums)]
    t_regions = {each_region:[] for each_region in t_regions}
    for cur_t, cur_y in zip(t.squeeze(), y.squeeze()):
        for cur_region in t_regions.keys():
            if (cur_region[0] <= cur_t) & (cur_t < cur_region[1]):
                t_regions[cur_region].append(cur_y)
            else:
                continue
    y_quantile = []
    for each_t in t.squeeze():
        for cur_region in t_regions.keys():
            if (cur_region[0] <= each_t) & (each_t < cur_region[1]):
                y_quantile.append(np.quantile(t_regions[cur_region], tau))
    # b-spline coefficient estimation
    result = []
    knots = np.array([np.min(t)] * k + list(np.linspace(np.min(t), np.max(t), dn + 1 - k)) + [np.max(t)] * k)
    B = BSpline.design_matrix(t.squeeze(), knots.squeeze(), k).toarray()
    for cur_x_id in tqdm(range(x.shape[1])):
        cur_x = x[:, cur_x_id].reshape((-1, 1))
        cur_x = np.hstack((B, B * cur_x))
        reg = QuantReg(y, cur_x).fit(q=tau, max_iter=2000)
        beta = reg.params.reshape((-1, 1))
        cur_f = np.dot(cur_x, beta) - np.array(y_quantile).reshape((-1, 1))
        result.append(np.mean(cur_f ** 2))
    return result


if __name__ == "__main__":
    np.random.seed(20230817)
    n = 10000
    x = np.random.randn(n, 2)
    t = np.random.randn(n, 1)
    y = (t ** 2) * x[:, 0].reshape((-1, 1))
    print(qnis(y, x, t, tau=0.9, width=0.25, k=2, dn=5))