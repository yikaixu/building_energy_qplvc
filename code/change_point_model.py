import numpy as np
import math
from scipy.stats import t
from statsmodels.regression.quantile_regression import QuantReg
from tqdm import tqdm
import warnings
warnings.simplefilter('once', category=UserWarning)


def positify(x):
    x[x < 0] = 0
    return x


def temp_transform(temp_list, cp_list, model):
    if model == "5p":
        cp_lower, cp_upper = cp_list[0], cp_list[1]
        temp_upper_list = [positify(temp - cp_upper) for temp in temp_list]
        temp_lower_list = [positify(cp_lower - temp) for temp in temp_list]
        constant_list = [np.ones_like(temp) for temp in temp_list]
        temp_list_transformed = []
        for each_upper, each_lower, each_constant in zip(temp_upper_list, temp_lower_list, constant_list):
            temp_list_transformed.append(np.hstack((each_constant, each_upper, each_lower)))
    elif model == "4p":
        cp = cp_list[0]
        temp_upper_list = [positify(temp - cp) for temp in temp_list]
        temp_lower_list = [positify(cp - temp) for temp in temp_list]
        constant_list = [np.ones_like(temp) for temp in temp_list]
        temp_list_transformed = []
        for each_upper, each_lower, each_constant in zip(temp_upper_list, temp_lower_list, constant_list):
            temp_list_transformed.append(np.hstack((each_constant, each_upper, each_lower)))
    elif model == "3p-H":
        cp = cp_list[0]
        temp_lower_list = [positify(cp - temp) for temp in temp_list]
        constant_list = [np.ones_like(temp) for temp in temp_list]
        temp_list_transformed = []
        for each_lower, each_constant in zip(temp_lower_list, constant_list):
            temp_list_transformed.append(np.hstack((each_constant, each_lower)))
    elif model == "3p-C":
        cp = cp_list[0]
        temp_upper_list = [positify(temp - cp) for temp in temp_list]
        constant_list = [np.ones_like(temp) for temp in temp_list]
        temp_list_transformed = []
        for each_upper, each_constant in zip(temp_upper_list, constant_list):
            temp_list_transformed.append(np.hstack((each_constant, each_upper)))
    else:
        constant_list = [np.ones_like(temp) for temp in temp_list]
        temp_list_transformed = []
        for each_temp, each_constant in zip(temp_list, constant_list):
            temp_list_transformed.append(np.hstack((each_constant, each_temp)))
    return temp_list_transformed


def list_vstack(list):
    result = list[0]
    for item in list[1:]:
        result = np.vstack((result, item))
    return result


def coef_sig_test(x, y, beta, alpha):
    n = len(y)
    p = x.shape[1]
    t_stat_alpha = t.ppf(1 - alpha * 0.5, n - p)
    cov_inv = np.linalg.inv(np.dot(x.T, x))
    sigma = np.linalg.norm(y - np.dot(x, beta)) / (n - p) ** 0.5
    t_stat_list = []
    for beta_id, each_beta in enumerate(beta):
        if beta_id == 0:
            continue
        t_stat_list.append(np.abs(each_beta / (sigma * cov_inv[beta_id, beta_id] ** 0.5)))
    return (np.array(t_stat_list) > t_stat_alpha).all()


def data_population_test(temp_transformed, prop, model):
    result = np.sum(temp_transformed != 0, axis=0) / len(temp_transformed)
    if model == '5p':
        result_middle = np.sum(temp_transformed[:, 1:], axis=1)
        result_middle = np.sum(result_middle == 0) / len(temp_transformed)
        return (result[1:] >= prop).all() and result_middle >= prop
    elif model == '4p':
        return (result[1:] >= prop).all()
    elif model in ['3p-H', '3p-C']:
        result_const = np.sum(temp_transformed == 0, axis=0) / len(temp_transformed)
        return (result[1:] >= prop).all() and (result_const[1:] >= prop).all()


def qs_score(y_true, y_pred, quantile_level):
    result = np.abs(y_true - y_pred)
    result[y_true < y_pred] = result[y_true < y_pred] * (1 - quantile_level)
    result[y_true >= y_pred] = result[y_true >= y_pred] * quantile_level
    return np.mean(result)


def cpm_fit(temp_list, y_list, lower_limit_temp, upper_limit_temp, step_size, tau):
    # -- fit 5p model
    y_total = list_vstack(y_list)
    qs_list = []
    beta_list = []
    temp_total_list = []
    cp_list = []
    temp_lower = math.ceil(lower_limit_temp + step_size)
    temp_upper = math.floor(upper_limit_temp - step_size)
    print("5p model...")
    for cp_upper in tqdm(np.arange(temp_lower, temp_upper, step_size)):
        for cp_lower in np.arange(temp_lower, cp_upper, step_size):
            temp_list_transformed = temp_transform(temp_list, [cp_lower, cp_upper], model="5p")
            temp_total = list_vstack(temp_list_transformed)
            reg = QuantReg(y_total, temp_total).fit(q=tau, max_iter=2000)
            beta_list.append(reg.params.reshape((-1, 1)))
            qs_list.append(qs_score(y_total, np.dot(temp_total, beta_list[-1]), tau))
            temp_total_list.append(temp_total)
            cp_list.append([cp_lower, cp_upper])
    beta = beta_list[np.argmin(np.array(qs_list))]
    temp_total = temp_total_list[np.argmin(np.array(qs_list))]
    cp = cp_list[np.argmin(np.array(qs_list))]
    # ---- 5p model test
    shape_test = beta[1] > 0 and beta[2] > 0
    coef_test = coef_sig_test(temp_total, y_total, beta, alpha=0.05)
    dp_test = data_population_test(temp_total, prop=0.1, model='5p')
    if shape_test[0] and coef_test and dp_test:
        return beta, cp, "5p"
    # -- fit 4p model
    print("4p model...")
    qs_list = []
    beta_list = []
    temp_total_list = []
    cp_list = []
    for cp in tqdm(np.arange(temp_lower, temp_upper, step_size)):
        temp_list_transformed = temp_transform(temp_list, [cp], model="4p")
        temp_total = list_vstack(temp_list_transformed)
        reg = QuantReg(y_total, temp_total).fit(q=tau, max_iter=2000)
        beta_list.append(reg.params.reshape((-1, 1)))
        qs_list.append(qs_score(y_total, np.dot(temp_total, beta_list[-1]), tau))
        temp_total_list.append(temp_total)
        cp_list.append([cp])
    beta = beta_list[np.argmin(np.array(qs_list))]
    temp_total = temp_total_list[np.argmin(np.array(qs_list))]
    cp = cp_list[np.argmin(np.array(qs_list))]
    # ---- 4p model test
    shape_test = (beta[1] > 0 and beta[2] > 0) or (beta[1] < 0 and beta[2] > 0) or (beta[1] > 0 and beta[2] < 0)
    coef_test = coef_sig_test(temp_total, y_total, beta, alpha=0.05)
    dp_test = data_population_test(temp_total, prop=0.1, model='4p')
    if shape_test[0] and coef_test and dp_test:
        return beta, cp, "4p"
    # -- fit 3p-H model
    print("3p-H model...")
    qs_list = []
    beta_list = []
    temp_total_list = []
    cp_list = []
    for cp in tqdm(np.arange(temp_lower, temp_upper, step_size)):
        temp_list_transformed = temp_transform(temp_list, [cp], model="3p-H")
        temp_total = list_vstack(temp_list_transformed)
        reg = QuantReg(y_total, temp_total).fit(q=tau, max_iter=2000)
        beta_list.append(reg.params.reshape((-1, 1)))
        qs_list.append(qs_score(y_total, np.dot(temp_total, beta_list[-1]), tau))
        temp_total_list.append(temp_total)
        cp_list.append([cp])
    beta = beta_list[np.argmin(np.array(qs_list))]
    temp_total = temp_total_list[np.argmin(np.array(qs_list))]
    cp = cp_list[np.argmin(np.array(qs_list))]
    # ---- 3p-H model test
    shape_test = beta[1] > 0
    coef_test = coef_sig_test(temp_total, y_total, beta, alpha=0.05)
    dp_test = data_population_test(temp_total, prop=0.1, model='3p-H')
    if shape_test[0] and coef_test and dp_test:
        return beta, cp, "3p-H"
    # -- fit 3p-C model
    print("3p-C model...")
    qs_list = []
    beta_list = []
    temp_total_list = []
    cp_list = []
    for cp in tqdm(np.arange(temp_lower, temp_upper, step_size)):
        temp_list_transformed = temp_transform(temp_list, [cp], model="3p-C")
        temp_total = list_vstack(temp_list_transformed)
        reg = QuantReg(y_total, temp_total).fit(q=tau, max_iter=2000)
        beta_list.append(reg.params.reshape((-1, 1)))
        qs_list.append(qs_score(y_total, np.dot(temp_total, beta_list[-1]), tau))
        temp_total_list.append(temp_total)
        cp_list.append([cp])
    beta = beta_list[np.argmin(np.array(qs_list))]
    temp_total = temp_total_list[np.argmin(np.array(qs_list))]
    cp = cp_list[np.argmin(np.array(qs_list))]
    # ---- 3p-C model test
    shape_test = beta[1] > 0
    coef_test = coef_sig_test(temp_total, y_total, beta, alpha=0.05)
    dp_test = data_population_test(temp_total, prop=0.1, model='3P-C')
    if shape_test[0] and coef_test and dp_test:
        return beta, cp, "3p-C"
    # -- fit 2p model
    print("2p model...")
    temp_list_transformed = temp_transform(temp_list, [], model="2p")
    temp_total = list_vstack(temp_list_transformed)
    reg = QuantReg(y_total, temp_total).fit(q=tau, max_iter=2000)
    beta = reg.params.reshape((-1, 1))
    return beta, [], "2p"


def cpm_predict(temp_list, beta, cp_list, model):
    temp_pred = list_vstack(temp_transform(temp_list, cp_list, model=model))
    return np.dot(temp_pred, beta)









