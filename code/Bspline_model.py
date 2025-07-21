import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from scipy.interpolate import BSpline


def list_vstack(list):
    result = list[0]
    for item in list[1:]:
        result = np.vstack((result, item))
    return result


def Bspline_fit(t_list, y_list, k, dn, tau):
    t = list_vstack(t_list)
    y = list_vstack(y_list)
    knots = np.array([np.min(t)] * k + list(np.linspace(np.min(t), np.max(t), dn + 1 - k)) + [np.max(t)] * k)
    B = BSpline.design_matrix(t.squeeze(), knots.squeeze(), k).toarray()
    reg = QuantReg(y, B).fit(q=tau, max_iter=2000)
    return reg.params.reshape((-1, 1)), knots


def Bspline_predict(t_list, k, beta, knots):
    t = list_vstack(t_list)
    B = BSpline.design_matrix(t.squeeze(), knots.squeeze(), k).toarray()
    return np.dot(B, beta)








