import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from scipy.interpolate import BSpline


def list_vstack(data_list):
    result = data_list[0]
    for item in data_list[1:]:
        result = np.vstack((result, item))
    return result


def list_hstack(data_list):
    result = data_list[0]
    for item in data_list[1:]:
        result = np.hstack((result, item))
    return result


def QVC_fit(x_list, t_list, y_list, dn_list, k, tau):
    x = list_vstack(x_list)
    t = list_vstack(t_list)
    y = list_vstack(y_list)
    knots = [np.array([np.min(t)] * k + list(np.linspace(np.min(t), np.max(t), each_dn + 1 - k)) + [np.max(t)] * k) for each_dn in dn_list]
    B_list = [cur_x.reshape((-1, 1)) * BSpline.design_matrix(t.squeeze(), cur_knot.squeeze(), k).toarray() for cur_x, cur_knot in zip(x.T, knots)]
    x_total = list_hstack(B_list)
    reg = QuantReg(y, x_total).fit(q=tau, max_iter=2000)
    return reg.params.reshape((-1, 1)), knots


def QVC_predict(x_list, t_list, k, beta, knots):
    x = list_vstack(x_list)
    t = list_vstack(t_list)
    B_list = [cur_x.reshape((-1, 1)) * BSpline.design_matrix(t.squeeze(), cur_knot.squeeze(), k).toarray() for cur_x, cur_knot in zip(x.T, knots)]
    x_total = list_hstack(B_list)
    return np.dot(x_total, beta), t





