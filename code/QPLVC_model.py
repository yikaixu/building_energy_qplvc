import numpy as np
# from sklearn.linear_model import QuantileRegressor
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


def QPLVC_fit(x_v_list, x_c_list, t_list, y_list, dn_list, k, tau):
    x_v = list_vstack(x_v_list)
    x_c = list_vstack(x_c_list)
    t = list_vstack(t_list)
    y = list_vstack(y_list)
    knots = [np.array([np.min(t)] * k + list(np.linspace(np.min(t), np.max(t), each_dn + 1 - k)) + [np.max(t)] * k) for each_dn in dn_list]
    B_v_list = [cur_x_v.reshape((-1, 1)) * BSpline.design_matrix(t.squeeze(), cur_knot.squeeze(), k).toarray() for cur_x_v, cur_knot in zip(x_v.T, knots)]
    B_v = list_hstack(B_v_list)
    x_total = np.hstack((B_v, x_c))
    # ----- solve using sklearn -----
    # reg = QuantileRegressor(quantile = tau, alpha = 0, fit_intercept = False, solver = "revised simplex").fit(x_total, y_total)
    # return reg.coef_
    # ----- solve using statsmodels -----
    reg = QuantReg(y, x_total).fit(q=tau, max_iter=2000)
    return reg.params.reshape((-1, 1)), knots


def QPLVC_predict(x_v_list, x_c_list, t_list, k, beta, knots):
    x_v = list_vstack(x_v_list)
    x_c = list_vstack(x_c_list)
    t = list_vstack(t_list)
    B_v_list = [cur_x_v.reshape((-1, 1)) * BSpline.design_matrix(t.squeeze(), cur_knot.squeeze(), k).toarray() for cur_x_v, cur_knot in zip(x_v.T, knots)]
    B_v = list_hstack(B_v_list)
    x_total = np.hstack((B_v, x_c))
    return np.dot(x_total, beta), t




