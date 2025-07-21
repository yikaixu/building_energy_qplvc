import contextlib
import os
import sys
import shap
# shap.initjs()
import timeit
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import seaborn as sns
from QPLVC_model import QPLVC_fit, QPLVC_predict
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import make_scorer
from pprint import pprint
from scipy.interpolate import BSpline
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from sklearn_quantile import SampleRandomForestQuantileRegressor
from sklearn.utils import parallel_backend
from QNIS import qnis


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Monkey patch the fit method of your estimator to silence it
original_fit = SampleRandomForestQuantileRegressor.fit

def silent_fit(self, *args, **kwargs):
    with suppress_stdout_stderr():
        return original_fit(self, *args, **kwargs)

SampleRandomForestQuantileRegressor.fit = silent_fit


def load_data(test_year, energy_type):
    meta_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "data" + os.sep + "real_data_raw"
    meta_data = pd.read_excel(meta_data_path + os.sep + "5 train_building_info.xlsx")
    train_data_list = []
    train_bid_list = []
    test_data_list = []
    test_bid_list = []
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "processed_realdata"
    for each_file in os.listdir(data_path):
        cur_data = pd.read_csv(data_path + os.sep + each_file)
        cur_bid = each_file.split('.')[0].split('_')[0]
        cur_energy_type = each_file.split('.')[0].split('_')[1]
        if cur_energy_type == energy_type:
            cur_year = each_file.split('.')[0].split('_')[-1]
            if cur_year == test_year:
                test_data_list.append(cur_data)
                test_bid_list.append(cur_bid)
            else:
                train_data_list.append(cur_data)
                train_bid_list.append(cur_bid)
    return train_data_list, train_bid_list, test_data_list, test_bid_list, meta_data


def time_process(data):
    # transform datetime to integer index
    for each_data in data:
        each_data.drop(each_data[each_data['Time'] == 'NaT'].index, inplace=True)
        each_data['is_weekday'] = [int(each_hour.weekday() < 5) for each_hour in pd.to_datetime(each_data['Time']).tolist()]
        each_data['Time'] = np.arange(len(each_data))
    return data


def hvactype_coding(cur_type, ref_type_list):
    if cur_type in ref_type_list:
        return 1
    else:
        return 0


def amplify_meta_data(meta_data, weather_data_list, bid_List):
    meta_data_list = []
    for cur_bid, cur_weather_data in zip(bid_List, weather_data_list):
        cur_meta = meta_data[meta_data["BuildingID"] == int(cur_bid)]
        meta_data_list.append(pd.concat([cur_meta.iloc[:, 1:]] * len(cur_weather_data), ignore_index = True))
    return meta_data_list


# -- discard
def produce_cross_covariate(weather_data_list, meta_data_list, vary_cov):
    weather_predictors = weather_data_list[0].columns.tolist()
    weather_predictors = [item for item in weather_predictors if item not in ['Record', vary_cov]]
    meta_predictors = meta_data_list[0].columns.tolist()
    result = weather_data_list.copy()
    for each_weather_predictor in weather_predictors:
        for each_meta_predictor in meta_predictors:
            cur_cross_predictor = "{0} * {1}".format(each_weather_predictor, each_meta_predictor)
            for each_meta_data, each_result_data in zip(meta_data_list, result):
                each_result_data[cur_cross_predictor] = each_result_data[each_weather_predictor] * each_meta_data[each_meta_predictor]
    return result


def list_vstack(list):
    result = list[0]
    for item in list[1:]:
        result = np.vstack((result, item))
    return result


def qs_score(y_true, y_pred, quantile_level):
    result = np.abs(y_true - y_pred)
    result[y_true < y_pred] = result[y_true < y_pred] * (1 - quantile_level)
    result[y_true >= y_pred] = result[y_true >= y_pred] * quantile_level
    return np.mean(result)


def winkler_score(y_true, y_pred_l, y_pred_u, alpha):
    result = y_pred_u - y_pred_l
    result[y_true < y_pred_l] = result[y_true < y_pred_l] + 2 * (y_pred_l[y_true < y_pred_l] - y_true[y_true < y_pred_l]) / alpha
    result[y_true > y_pred_u] = result[y_true > y_pred_u] + 2 * (y_true[y_true > y_pred_u] - y_pred_u[y_true > y_pred_u]) / alpha
    return np.mean(result)


def coverage_probability(y_true, y_pred_l, y_pred_u):
    return np.sum((y_pred_l <= y_true) & (y_pred_u >= y_true)) / len(y_true)


def print_result_qs(QPLVC_qs_list, QRF_qs_list, QGB_qs_list, energy_type):
    print(" ******* method: QPLVC; metric: qs (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QPLVC_qs_list, axis=0))
    # print(" ******* method: QPLVC; metric: qs (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QPLVC_qs_list, axis=0))

    print(" ******* method: QRF; metric: qs (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QRF_qs_list, axis=0))
    # print(" ******* method: QRF; metric: qs (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QRF_qs_list, axis=0))

    print(" ******* method: QGB; metric: qs (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QGB_qs_list, axis=0))
    # print(" ******* method: QGB; metric: qs (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QGB_qs_list, axis=0))


def print_result_ws_cp(QPLVC_ws_list, QPLVC_cp_list, QRF_ws_list, QRF_cp_list, QGB_ws_list, QGB_cp_list, energy_type):
    print(" ******* method: QPLVC; metric: ws (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QPLVC_ws_list, axis=0))
    # print(" ******* method: QPLVC; metric: ws (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QPLVC_ws_list, axis=0))
    print(" ******* method: QPLVC; metric: cp (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QPLVC_cp_list, axis=0))
    # print(" ******* method: QPLVC; metric: cp (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QPLVC_cp_list, axis=0))

    print(" ******* method: QRF; metric: ws (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QRF_ws_list, axis=0))
    # print(" ******* method: QRF; metric: ws (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QRF_ws_list, axis=0))
    print(" ******* method: QRF; metric: cp (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QRF_cp_list, axis=0))
    # print(" ******* method: QRF; metric: cp (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QRF_cp_list, axis=0))

    print(" ******* method: QGB; metric: ws (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QGB_ws_list, axis=0))
    # print(" ******* method: QGB; metric: ws (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QGB_ws_list, axis=0))
    print(" ******* method: QGB; metric: cp (mean); energy type:{0} ******** ".format(energy_type))
    print(np.mean(QGB_cp_list, axis=0))
    # print(" ******* method: QGB; metric: cp (std); energy type:{0} ******** ".format(energy_type))
    # print(np.std(QGB_cp_list, axis=0))


def add_ar_predictor(data):
    for each_data in data:
        each_data['oneday_ahead_record'] = each_data['Record'].to_list()[-24: ] + each_data['Record'].to_list()[: -24]
        #each_data['oneweek_ahead_record'] = each_data['Record'].to_list()[168: ] + each_data['Record'].to_list()[0: 168]
    return data


# -- discard
def get_scatter_plot(t, y_true, y_pred_lower, y_pred_upper, cur_q_lower, cur_q_upper, cur_method, energy_type, vary_cov, cur_year):
    fig, ax = plt.subplots()
    # plt.scatter(t, y_true, marker='o', color='red', s=0.5, label='true', alpha=0.1)
    plt.scatter(t, y_pred_lower, marker='o', color='blue', s=0.5, label='{0}-q predict'.format(cur_q_lower), alpha=0.1)
    plt.scatter(t, y_pred_upper, marker='o', color='green', s=0.5, label='{0}-q predict'.format(cur_q_upper), alpha=0.1)
    # red_patch = mpatches.Patch(color='red', label='true')
    blue_patch = mpatches.Patch(color='blue', label='{0}-q predict'.format(cur_q_lower))
    green_patch = mpatches.Patch(color='green', label='{0}-q predict'.format(cur_q_upper))
    ax.legend(handles=[blue_patch, green_patch], loc='upper right')
    plt.xlabel(vary_cov)
    plt.ylabel("Record")
    plt.ylim((0, 3000))
    plt.savefig("{0}_{1}_{2}scatter_{3}.png".format(energy_type, cur_method, cur_q_upper - cur_q_lower, cur_year))


# -- discard
def get_pi_plot(y_true, y_pred_med, y_pred_lower, y_pred_upper, cur_q_lower, cur_q_upper, cur_method, energy_type, cur_year):
    fig, ax = plt.subplots()
    plt.scatter(y_pred_med, y_true, marker='o', color='red', s=0.5, label='true', alpha=0.1)
    plt.scatter(y_pred_med, y_pred_lower, marker='o', color='blue', s=0.5, label='{0}-q predict'.format(cur_q_lower), alpha=0.1)
    plt.scatter(y_pred_med, y_pred_upper, marker='o', color='green', s=0.5, label='{0}-q predict'.format(cur_q_upper), alpha=0.1)
    plt.plot(np.linspace(0, 3500, 100), np.linspace(0, 3500, 100), linestyle='-', color='gray')
    red_patch = mpatches.Patch(color='red', label='true')
    blue_patch = mpatches.Patch(color='blue', label='{0}-q predict'.format(cur_q_lower))
    green_patch = mpatches.Patch(color='green', label='{0}-q predict'.format(cur_q_upper))
    ax.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right')
    plt.xlabel('predicted conditional median')
    plt.ylabel("observed values")
    plt.ylim((0, 3500))
    plt.xlim((0, 3500))
    plt.savefig("{0}_{1}_{2}PI_bar{3}.png".format(energy_type, cur_method, cur_q_upper - cur_q_lower, cur_year))


# -- discard
def get_pi_plot_bar(y_true, y_pred_med, y_pred_lower, y_pred_upper, cur_q_lower, cur_q_upper, cur_method, energy_type, cur_year):
    fig, ax = plt.subplots()
    inside = ((y_pred_upper - y_true) >= 0) & ((y_true - y_pred_lower) >= 0)
    outside = np.logical_not(inside)
    plt.scatter(y_pred_med[inside], y_true[inside], marker='o', color='blue', s=0.5, label='true', alpha=0.1)
    plt.scatter(y_pred_med[outside], y_true[outside], marker='o', color='red', s=0.5, label='true', alpha=0.1)
    delta = y_pred_upper.squeeze() - y_pred_lower.squeeze()
    plt.errorbar(y_pred_med.squeeze(), y_pred_lower.squeeze(), yerr=(len(y_true) * [0], delta), fmt='o', elinewidth=0.5, color='gray', alpha=0.003)
    plt.plot(np.linspace(0, 3500, 100), np.linspace(0, 3500, 100), linestyle='-', color='black')
    # red_patch = mpatches.Patch(color='red', label='true')
    # blue_patch = mpatches.Patch(color='blue', label='{0}-q predict'.format(cur_q_lower))
    # green_patch = mpatches.Patch(color='green', label='{0}-q predict'.format(cur_q_upper))
    # ax.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right')
    plt.xlabel("predicted conditional median values")
    plt.ylabel("observed values")
    plt.ylim((0, 3500))
    plt.xlim((0, 3500))
    plt.savefig("{0}_{1}_{2}PI_bar_{3}.png".format(energy_type, cur_method, cur_q_upper - cur_q_lower, cur_year))


def get_vary_coef_plot(t, beta_list, knots_list, vary_cov, energy_type, visualize_q, feature_names, feature_names_all):
    t_plot = np.linspace(np.min(t), np.max(t), 8760).reshape((-1, 1))
    cov_id_list = [feature_names_all.index(item) for item in feature_names]
    fig, ax = plt.subplots()
    max_list = []
    min_list = []
    for cov_id, cov, color in zip(cov_id_list, feature_names, ['green', 'red', 'blue']):
        cur_beta_id = np.arange(cov_id * dn, (cov_id + 1) * dn)
        coef_plot, _ = QPLVC_predict([np.ones_like(t_plot)], [np.zeros_like(t_plot)], [t_plot], k, np.vstack((beta_list[cur_beta_id], np.array([[0]]))), [knots_list[cov_id]])
        max_list.append(np.max(coef_plot))
        min_list.append(np.min(coef_plot))
        if cov == 'hvac_jizhong':
            plt.plot(t_plot, coef_plot, linestyle="-", label='central_HVAC', color=color)
        else:
            plt.plot(t_plot, coef_plot, linestyle="-", label=cov, color=color)
    plt.legend(loc="upper left")
    plt.xlabel("air temperature (\u00B0C)")
    plt.ylabel("coefficient value")
    axis_min = np.array(min_list).min()
    axis_max = np.array(max_list).max()
    plt.ylim((axis_min, axis_max))
    plt.grid()
    if vary_cov == 'Time':
        coverage_id = (t_plot >= 1416) & (t_plot <= 3623)
        plt.fill_between(t_plot.squeeze(), y1=axis_min * 0.9, y2=axis_max * 1.1, where=coverage_id.squeeze(), facecolor='lightcyan')
        plt.text(1416, axis_min + 0.5, 'Mar-May')
        coverage_id = (t_plot >= 5832) & (t_plot <= 8015)
        plt.fill_between(t_plot.squeeze(), y1=axis_min * 0.9, y2=axis_max * 1.1, where=coverage_id.squeeze(), facecolor='lightcyan')
        plt.text(5823, axis_min + 0.5, 'Sep-Nov')
    plt.savefig("coef_{0}_{1}.pdf".format(energy_type, visualize_q))


def smooth_mv(df, sort_feature, slide_width=50):
    feature_sort = df.sort_values(by=[sort_feature]).loc[:, sort_feature].values
    shap_sort = df.sort_values(by=[sort_feature]).loc[:, 'shap'].values
    x_plot = []
    y_plot = []
    cur_id = 0
    while cur_id + slide_width < len(feature_sort):
        x_plot.append(feature_sort[cur_id])
        y_plot.append(np.mean(shap_sort[cur_id: cur_id + slide_width]))
        cur_id += slide_width
    return x_plot, y_plot


def get_shap_plot(model, data, energy_type, cur_q, plot_features, vary_cov, cur_mean_std):
    nl_mean, nl_std, _, _ = cur_mean_std
    explainer = shap.Explainer(model)
    shap_raw = explainer(data)
    shap_df = pd.DataFrame(shap_raw.values, columns=shap_raw.feature_names, index=data.index)
    feature_names = data.columns.tolist()
    for feature in plot_features:
        cur_fid = feature_names.index(feature)
        cur_mean = nl_mean.squeeze()[cur_fid]
        cur_std = nl_std.squeeze()[cur_fid]
        fig, ax = plt.subplots()
        # -- scatter
        plt.scatter(data.loc[:, feature] * cur_std + cur_mean, shap_df.loc[:, feature], c=data.loc[:, vary_cov], cmap='viridis', alpha=0.2, s=1.1)
        plt.ylabel('SHAP value (kWh)')
        plt.xlabel(feature.replace('_', ' '))
        clb = plt.colorbar()
        clb.ax.set_title("air temperature (\u00B0C)")
        if feature == 'Area':
            plt.xlabel("Area (m\u00b2)")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        elif feature == 'hvac_jizhong':
            plt.xlabel("central_HVAC")
        else:
            plt.xlabel(feature.replace('_', ' '))
        if feature == 'Stair2':
            plt.xticks([0, 1, 2, 3, 4])
        elif feature == 'hvac_jizhong':
            plt.xticks([0, 1])

        # # -- line
        # t_upper = np.quantile(data.loc[:, vary_cov].values, 0.8)
        # t_lower = np.quantile(data.loc[:, vary_cov].values, 0.2)
        # shap_upper = shap_df[data[vary_cov] > t_upper].loc[:, feature].values
        # feature_upper = data[data[vary_cov] > t_upper].loc[:, feature].values
        # cur_df_shap_feature = pd.DataFrame(data=np.hstack((feature_upper.reshape((-1, 1)), shap_upper.reshape((-1, 1)))), columns=[feature, 'shap'])
        # x_plot, y_plot = smooth_mv(cur_df_shap_feature, feature)
        # plt.plot(np.array(x_plot) * cur_std + cur_mean, y_plot, linestyle='--', label='high air temperature group', color='red', linewidth=0.7)
        # shap_lower = shap_df[data[vary_cov] < t_lower].loc[:, feature].values
        # feature_lower = data[data[vary_cov] < t_lower].loc[:, feature].values
        # cur_df_shap_feature = pd.DataFrame(data=np.hstack((feature_lower.reshape((-1, 1)), shap_lower.reshape((-1, 1)))), columns=[feature, 'shap'])
        # x_plot, y_plot = smooth_mv(cur_df_shap_feature, feature)
        # plt.plot(np.array(x_plot) * cur_std + cur_mean, y_plot, linestyle='--', label='low air temperature group', color='blue', linewidth=0.7)
        # plt.legend()
        plt.savefig("{0}_{1}_{2}_dependence.png".format(energy_type, feature, cur_q))


def print_runtime(qplvc, qrf, qgb):
    for item in ['train', 'test']:
        print('***** {0} time *****'.format(item))
        print('qplvc:{0}({1})'.format(np.mean(np.array(qplvc[item])), np.std(np.array(qplvc[item]))))
        print('qrf:{0}({1})'.format(np.mean(np.array(qrf[item])), np.std(np.array(qrf[item]))))
        print('qgb:{0}({1})'.format(np.mean(np.array(qgb[item])), np.std(np.array(qgb[item]))))


# discard
def print_data_sumarize(train_x, train_meta, train_y, test_x, test_meta, test_y, nl_predictors, l_predictors, energy_type):
    x_df = pd.DataFrame(data=np.vstack((list_vstack(train_x), list_vstack(test_x))), columns=nl_predictors)
    x_df_numc = x_df.loc[:, ['Time', 'air temperature', 'dew temperature', 'relative humidity', 'pressure', 'wind speed', 'oneday_ahead_record']]
    meta_df = pd.DataFrame(data=np.vstack((list_vstack(train_meta), list_vstack(test_meta))), columns=l_predictors)
    meta_numc = meta_df.loc[:, ['Area*Stair1', 'Area*Stair2']]
    meta_cate = meta_df.loc[:, ['hvac_jizhong', 'hvac_fengji', 'hvac_else']]
    y_np = np.vstack((list_vstack(train_y), list_vstack(test_y)))
    print(" ***** Energy type: {0} *****".format(energy_type))
    print("* mean of x:{0};\n mean of meta:{1};\n mean of y:{2}\n".format(x_df_numc.apply(np.mean, axis=0), meta_numc.apply(np.mean, axis=0), np.mean(y_np)))
    print("* std of x:{0};\n std of meta:{1};\n std of y:{2}\n".format(x_df_numc.apply(np.std, axis=0), meta_numc.apply(np.std, axis=0), np.std(y_np)))
    print("* median of x:{0};\n median of meta:{1};\n median of y:{2}\n".format(x_df_numc.apply(np.median, axis=0), meta_numc.apply(np.median, axis=0), np.median(y_np)))
    print("* min of x:{0};\n min of meta:{1};\n min of y:{2}\n".format(x_df_numc.apply(np.min, axis=0), meta_numc.apply(np.min, axis=0), np.min(y_np)))
    print("* max of x:{0};\n max of meta:{1};\n max of y:{2}\n".format(x_df_numc.apply(np.max, axis=0), meta_numc.apply(np.max, axis=0), np.max(y_np)))
    print("* percentage of weekday:{0}\n".format(np.sum(x_df['is_weekday'].values) / len(x_df)))
    print("* percentage of meta:{0}\n".format(meta_cate.apply(np.sum, axis=0) / len(meta_cate)))

# discard
def separate_y_according_t(y, t, energy_type):
   result_y = []
   if energy_type == 'Q':
       mar_may_id = (t > 1416) & (t <= 3623)
       jun_aug_id = (t > 3623) & (t <= 5832)
       sep_nov_id = (t > 5832) & (t <= 8015)
       dec_feb_id = (t > 8015) | (t <= 1416)
       result_y.append(y[mar_may_id])
       result_y.append(y[jun_aug_id])
       result_y.append(y[sep_nov_id])
       result_y.append(y[dec_feb_id])
   elif energy_type == 'W':
        low_id = (t <= 5)
        mid_id = (t > 5) & (t <= 25)
        up_id = (t > 25)
        result_y.append(y[low_id])
        result_y.append(y[mid_id])
        result_y.append(y[up_id])
   return result_y


if __name__ == "__main__":
    # -- global parameters
    np.random.seed(20250419)
    quantile_level = [0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_level = [0.3, 0.2, 0.1]
    # quantile_level_grid_qplvc = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    quantile_level_grid_qplvc = [0.9] # use this for Figure 8
    qplvc_time = {'train': [], 'test': []}
    qrf_time = {'train': [], 'test': []}
    qgb_time = {'train': [], 'test': []}
    plot_t_group_labels = {'Q':['Mar-May', 'Jun-Aug', 'Sep-Nov', 'Dec-Feb'], 'W':['at<=5', '5<at<=25', 'at>25']}

    # -- parameters of QPLVC model
    k = 3
    dn = 11

    for energy_type in ['W']:
        vary_cov = 'air temperature'
        QPLVC_qs_list,  QRF_qs_list, QGB_qs_list = [], [], []
        QPLVC_ws_list,  QRF_ws_list, QGB_ws_list = [], [], []
        QPLVC_cp_list,  QRF_cp_list, QGB_cp_list = [], [], []
        for test_year in ['2017']:
            # -- load data
            train_data_list, train_bid_list, test_data_list, test_bid_list, meta_data = load_data(test_year, energy_type)

            # -- process weather data
            # ---- process: 'Time'
            train_data_list = time_process(train_data_list)
            test_data_list = time_process(test_data_list)
            # ---- process: 'Record'
            train_data_list = add_ar_predictor(train_data_list)
            test_data_list = add_ar_predictor(test_data_list)
            # ---- delete row contain missing
            train_data_list = [each_data.dropna(axis=0, how='any') for each_data in train_data_list]
            test_data_list = [each_data.dropna(axis=0, how='any') for each_data in test_data_list]
            # ---- delete outlier
            upper_bar = np.quantile(list_vstack([item['Record'].values.reshape((-1, 1)) for item in train_data_list]), 0.99999)
            train_data_list = [each_data[(each_data['Record'] >= 0) & (each_data['Record'] < upper_bar)].reset_index(drop=True) for each_data in train_data_list]
            train_data_list = [each_data[(each_data['oneday_ahead_record'] >= 0) & (each_data['oneday_ahead_record'] < upper_bar)].reset_index(drop=True) for each_data in train_data_list]
            upper_bar = np.quantile(list_vstack([item['Record'].values.reshape((-1, 1)) for item in test_data_list]), 0.99999)
            test_data_list = [each_data[(each_data['Record'] >= 0) & (each_data['Record'] < upper_bar)].reset_index(drop=True) for each_data in test_data_list]
            test_data_list = [each_data[(each_data['oneday_ahead_record'] >= 0) & (each_data['oneday_ahead_record'] < upper_bar)].reset_index(drop=True) for each_data in test_data_list]


            # -- process meta data
            # ---- process: 'HVACType'
            meta_data['hvac_jizhong'] = meta_data['HVACType'].apply(lambda x: hvactype_coding(x, ['集中式全空气系统']))
            meta_data['hvac_fengji'] = meta_data['HVACType'].apply(lambda x: hvactype_coding(x, ['风机盘管＋新风系统']))
            meta_data['hvac_else'] = meta_data['HVACType'].apply(lambda x: hvactype_coding(x, ['分体式空调或VRV的局部式机组系统', '其它']))
            meta_data.drop(['HVACType'], axis=1, inplace=True)
            # ---- amplify meta data
            meta_data_list = amplify_meta_data(meta_data, train_data_list, train_bid_list)
            meta_data_test_list = amplify_meta_data(meta_data, test_data_list, test_bid_list)


            # # -- QNIS feature selection
            # weather_features = train_data_list[0].columns.tolist()
            # weather_features.remove('Record')
            # meta_features = meta_data_list[0].columns.tolist()
            # t_total = list_vstack([each.loc[:, vary_cov].values.reshape((-1, 1)) for each in train_data_list])
            # y_total = list_vstack([each.loc[:, 'Record'].values.reshape((-1, 1)) for each in train_data_list])
            # x_total = list_vstack([np.hstack((each_weather.drop(['Record'], axis=1).values, each_meta.values)) for each_weather, each_meta in zip(train_data_list, meta_data_list)])
            # for tau in quantile_level:
            #     qnis_value = np.array(qnis(y_total, x_total, t_total, tau, width=1, k=3, dn=11))
            #     screen_nl_df = pd.DataFrame(data=np.array([weather_features + meta_features, qnis_value]).T, columns=['var', 'qnis_value_{0}'.format(energy_type)])
            #     np.random.shuffle(x_total)
            #     qnis_value_random = np.array(qnis(y_total, x_total, t_total, tau, width=1, k=3, dn=11))
            #     cur_max_threshold = int(np.array(qnis_value_random).max())
            #     cur_min_threshold = int(np.array(qnis_value_random).min())
            #     screen_nl_df.to_csv("screen_nl_{0}_{1}_min{2}_max{3}.csv".format(energy_type, tau, cur_min_threshold, cur_max_threshold), index=False)

            # -- features for QPLVC
            nl_predictors_Q_one = ['oneday_ahead_record', 'Area', 'hvac_jizhong', 'hvac_fengji', 'Stair1', 'Stair2']
            nl_predictors_Q_three = ['Area', 'hvac_jizhong', 'Stair2', 'hvac_fengji', 'dew temperature', 'oneday_ahead_record']
            nl_predictors_Q_five = ['Area', 'relative humidity', 'dew temperature', 'pressure', 'oneday_ahead_record', 'is_weekday']
            nl_predictors_Q_seven = ['Stair1', 'Time', 'wind speed', 'dew temperature', 'hvac_else', 'relative humidity']
            nl_predictors_Q_nine = ['dew temperature', 'Stair1', 'hvac_jizhong', 'wind speed',  'Stair2', 'hvac_else']

            nl_predictors_W_one = ['oneday_ahead_record', 'is_weekday', 'hvac_jizhong', 'Stair1', 'hvac_fengji', 'Stair2']
            nl_predictors_W_three = ['Area', 'hvac_else', 'oneday_ahead_record', 'wind speed', 'is_weekday', 'Time']
            nl_predictors_W_five = ['hvac_jizhong', 'oneday_ahead_record', 'hvac_else', 'wind speed', 'Stair2', 'is_weekday']
            nl_predictors_W_seven = ['Stair1', 'Stair2', 'oneday_ahead_record', 'Area', 'hvac_else', 'hvac_jizhong']
            # nl_predictors_W_nine = ['Area', 'wind speed', 'relative humidity', 'Stair2', 'oneday_ahead_record', 'Stair1']
            nl_predictors_W_nine = ['Area', 'wind speed', 'relative humidity', 'Stair2', 'oneday_ahead_record', 'Stair1', 'hvac_jizhong', 'dew temperature', 'Time', 'pressure', 'hvac_else', 'hvac_fengji', 'is_weekday'] # for Figure 8

            l_predictors_Q_one = ['is_weekday', 'hvac_else', 'pressure', 'relative humidity', 'dew temperature', 'wind speed', 'Time']
            l_predictors_Q_three = ['is_weekday', 'wind speed', 'Stair1', 'Time', 'pressure', 'relative humidity', 'hvac_else']
            l_predictors_Q_five = ['Time', 'hvac_jizhong', 'wind speed', 'hvac_else', 'hvac_fengji', 'Stair1', 'Stair2']
            l_predictors_Q_seven = ['is_weekday', 'hvac_jizhong', 'Area', 'Stair2', 'hvac_fengji', 'oneday_ahead_record', 'pressure']
            l_predictors_Q_nine = ['relative humidity', 'hvac_fengji', 'is_weekday', 'Area', 'oneday_ahead_record', 'Time', 'pressure']

            l_predictors_W_one = ['dew temperature', 'Area', 'relative humidity', 'hvac_else', 'pressure', 'wind speed', 'Time']
            l_predictors_W_three = ['Stair1', 'dew temperature', 'hvac_fengji', 'Stair2', 'hvac_jizhong', 'relative humidity', 'pressure']
            l_predictors_W_five = ['dew temperature', 'pressure', 'relative humidity', 'Stair1', 'Time', 'hvac_fengji', 'Area']
            l_predictors_W_seven = ['wind speed', 'hvac_fengji', 'is_weekday', 'Time', 'dew temperature', 'relative humidity', 'pressure']
            # l_predictors_W_nine = ['hvac_jizhong', 'dew temperature', 'Time', 'pressure', 'hvac_else', 'hvac_fengji', 'is_weekday']
            l_predictors_W_nine = [] # For Figure 8


            qauntile2predictors_dict = {'Q': {0.05: [nl_predictors_Q_one, l_predictors_Q_one],
                                        0.1: [nl_predictors_Q_one, l_predictors_Q_one],
                                        0.15: [nl_predictors_Q_one, l_predictors_Q_one],
                                        0.2: [nl_predictors_Q_one, l_predictors_Q_one],
                                        0.25: [nl_predictors_Q_three, l_predictors_Q_three],
                                        0.3: [nl_predictors_Q_three, l_predictors_Q_three],
                                        0.35: [nl_predictors_Q_three, l_predictors_Q_three],
                                        0.4: [nl_predictors_Q_three, l_predictors_Q_three],
                                        0.45: [nl_predictors_Q_five, l_predictors_Q_five],
                                        0.5: [nl_predictors_Q_five, l_predictors_Q_five],
                                        0.55: [nl_predictors_Q_five, l_predictors_Q_five],
                                        0.6: [nl_predictors_Q_five, l_predictors_Q_five],
                                        0.65: [nl_predictors_Q_seven, l_predictors_Q_seven],
                                        0.7: [nl_predictors_Q_seven, l_predictors_Q_seven],
                                        0.75: [nl_predictors_Q_seven, l_predictors_Q_seven],
                                        0.8: [nl_predictors_Q_seven, l_predictors_Q_seven],
                                        0.85: [nl_predictors_Q_nine, l_predictors_Q_nine],
                                        0.9: [nl_predictors_Q_nine, l_predictors_Q_nine],
                                        0.95: [nl_predictors_Q_nine, l_predictors_Q_nine],
                                        },
                                        'W': {0.05: [nl_predictors_W_one, l_predictors_W_one],
                                        0.1: [nl_predictors_W_one, l_predictors_W_one],
                                        0.15: [nl_predictors_W_one, l_predictors_W_one],
                                        0.2: [nl_predictors_W_one, l_predictors_W_one],
                                        0.25: [nl_predictors_W_three, l_predictors_W_three],
                                        0.3: [nl_predictors_W_three, l_predictors_W_three],
                                        0.35: [nl_predictors_W_three, l_predictors_W_three],
                                        0.4: [nl_predictors_W_three, l_predictors_W_three],
                                        0.45: [nl_predictors_W_five, l_predictors_W_five],
                                        0.5: [nl_predictors_W_five, l_predictors_W_five],
                                        0.55: [nl_predictors_W_five, l_predictors_W_five],
                                        0.6: [nl_predictors_W_five, l_predictors_W_five],
                                        0.65: [nl_predictors_W_seven, l_predictors_W_seven],
                                        0.7: [nl_predictors_W_seven, l_predictors_W_seven],
                                        0.75: [nl_predictors_W_seven, l_predictors_W_seven],
                                        0.8: [nl_predictors_W_seven, l_predictors_W_seven],
                                        0.85: [nl_predictors_W_nine, l_predictors_W_nine],
                                        0.9: [nl_predictors_W_nine, l_predictors_W_nine],
                                        0.95: [nl_predictors_W_nine, l_predictors_W_nine],
                                        }}


            def normalize(data_list):
                data_all = list_vstack(data_list)
                mean_all = np.mean(data_all, axis=0).reshape((1, -1))
                std_all = np.std(data_all, axis=0).reshape((1, -1))
                return [(item - mean_all) / std_all for item in data_list], mean_all, std_all


            def data_process_train(nl_predictors, l_predictors, data_list, meta_list):
                # -- select modelling predictors and separate into x, t, y
                data_all_list = [pd.concat([each_data, each_meta], axis=1) for each_data, each_meta in zip(data_list, meta_list)]
                y = [item['Record'].values.reshape((-1, 1)) for item in data_all_list]
                t = [item[vary_cov].values.reshape((-1, 1)) for item in data_all_list]
                x_nl = [item.loc[:, nl_predictors].values for item in data_all_list]
                x_l = [item.loc[:, l_predictors].values for item in data_all_list]
                # normalize x
                x_nl, x_nl_mean, x_nl_std = normalize(x_nl)
                x_l, x_l_mean, x_l_std = normalize(x_l)
                # add constant
                x_nl = [np.hstack((item, np.ones(len(item)).reshape((-1, 1)))) for item in x_nl]
                return x_nl, x_nl_mean, x_nl_std, x_l, x_l_mean, x_l_std, y, t



            def data_process_test(nl_predictors, l_predictors, data_list, meta_list, nl_mean, nl_std, l_mean, l_std, t_train_max, t_train_min):
                # -- select modelling predictors and separate into x, t, y
                data_all_list = [pd.concat([each_data, each_meta], axis=1) for each_data, each_meta in zip(data_list, meta_list)]
                y = [item['Record'].values.reshape((-1, 1)) for item in data_all_list]
                t = [item[vary_cov].values.reshape((-1, 1)) for item in data_all_list]
                x_nl = [item.loc[:, nl_predictors].values for item in data_all_list]
                x_l = [item.loc[:, l_predictors].values for item in data_all_list]
                # ---- delete records t not in the range of training set
                retain_id_list = [((each >= t_train_min) & (each <= t_train_max)).squeeze() for each in t]
                y = [each[retain_id_list[id]] for id, each in enumerate(y)]
                t = [each[retain_id_list[id]] for id, each in enumerate(t)]
                x_nl = [each[retain_id_list[id]] for id, each in enumerate(x_nl)]
                x_l = [each[retain_id_list[id]] for id, each in enumerate(x_l)]
                # ---- normalize x and add constant
                x_nl = [(each - nl_mean) / nl_std for each in x_nl]
                x_l = [(each - l_mean) / l_std for each in x_l]
                x_nl = [np.hstack((item, np.ones(len(item)).reshape((-1, 1)))) for item in x_nl]
                return x_nl, x_l, y, t


        # -- Evaluation: QS
            # ---- prepare data
            train_data_of_quantile_levels = []
            train_mean_std_of_quantile_levels = []
            test_data_of_quantile_levels = []
            for each_q in quantile_level_grid_qplvc:
                cur_nl_predictors = qauntile2predictors_dict[energy_type][each_q][0]
                cur_l_predictors = qauntile2predictors_dict[energy_type][each_q][1]
                x_train_nl, x_nl_mean, x_nl_std, x_train_l, x_l_mean, x_l_std, y_train, t_train = data_process_train(cur_nl_predictors, cur_l_predictors, train_data_list, meta_data_list)
                train_mean_std_of_quantile_levels.append([x_nl_mean, x_nl_std, x_l_mean, x_l_std])
                cur_t_train_max = list_vstack(t_train).max()
                cur_t_train_min = list_vstack(t_train).min()
                x_test_nl, x_test_l, y_test, t_test = data_process_test(cur_nl_predictors, cur_l_predictors, test_data_list, meta_data_test_list, x_nl_mean, x_nl_std, x_l_mean, x_l_std, cur_t_train_max, cur_t_train_min)
                train_data_of_quantile_levels.append((x_train_nl, x_train_l, y_train, t_train))
                test_data_of_quantile_levels.append((x_test_nl, x_test_l, y_test, t_test))

            # # ---- QPLVC
            # qplvc_beta_knot_dict = {}
            # qplvc_pred_list = []
            # cur_year_QPLVC_qs_list = []
            # beta_list = []
            # knots_list = []
            # # ------ QPLVC model fit
            # for cur_data, each_q in zip(train_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_train_nl, x_train_l, y_train, t_train = cur_data
            #     dim = x_train_nl[0].shape[1]
            #     dn_list = [dn] * dim
            #     print("begin fit {0}-QPLVC...".format(each_q))
            #     start = timeit.default_timer()
            #     beta_qplvc, knots_qplvc = QPLVC_fit(x_train_nl, x_train_l, t_train, y_train, dn_list, k, each_q)
            #     stop = timeit.default_timer()
            #     qplvc_time['train'].append(stop - start)
            #     qplvc_beta_knot_dict['{0}'.format(each_q)] = {'coef': beta_qplvc, 'knot': knots_qplvc}
            # # ------ QPLVC model predict
            # for cur_data, each_q in zip(test_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_test_nl, x_test_l, y_test, t_test = cur_data
            #     cur_beta = qplvc_beta_knot_dict['{0}'.format(each_q)]['coef']
            #     cur_knot = qplvc_beta_knot_dict['{0}'.format(each_q)]['knot']
            #     start = timeit.default_timer()
            #     cur_pred, _ = QPLVC_predict(x_test_nl, x_test_l, t_test, k, cur_beta, cur_knot)
            #     stop = timeit.default_timer()
            #     qplvc_time['test'].append(stop - start)
            #     qplvc_pred_list.append(np.max(np.hstack((cur_pred, np.zeros_like(cur_pred))), axis=1).reshape((-1, 1)))
            # for cur_q in quantile_level:
            #     y_pred_qplvc = np.quantile(np.array(qplvc_pred_list).squeeze(), cur_q, axis=0).reshape((-1, 1))
            #     cur_year_QPLVC_qs_list.append(qs_score(list_vstack(y_test), y_pred_qplvc, cur_q))
            # QPLVC_qs_list.append(cur_year_QPLVC_qs_list)
            # # ------ visualization: coefficient value
            # visualize_q = 0.9
            # visualize_beta = qplvc_beta_knot_dict['{0}'.format(visualize_q)]['coef']
            # visualize_knot = qplvc_beta_knot_dict['{0}'.format(visualize_q)]['knot']
            # visualize_nl_predictors = qauntile2predictors_dict[energy_type][visualize_q][0]
            # get_vary_coef_plot(list_vstack(t_train), visualize_beta, visualize_knot, vary_cov, energy_type, visualize_q, ['Area', 'Stair2', 'hvac_jizhong'], visualize_nl_predictors)


            # # ---- QRF
            # cur_year_QRF_qs_list = []
            # param_grid = {"max_depth": [2, 5, 8], "min_samples_leaf": [1, 20, 40], "min_samples_split": [2, 20, 40]}
            # QRF_model_dict = {}
            # QRF_pred_list = []
            # # ------ QRF model fit
            # for cur_data, each_q in zip(train_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_train_nl, x_train_l, y_train, t_train = cur_data
            #     x_train = [np.hstack((each_x_nl, each_x_l, each_t))  for each_x_nl, each_x_l, each_t in zip(x_train_nl, x_train_l, t_train)]
            #     qrf = SampleRandomForestQuantileRegressor(random_state=0, q=each_q)
            #     cur_scorer = make_scorer(mean_pinball_loss, alpha=each_q, greater_is_better=False)
            #     print("begin fit {0}-QRF...".format(each_q))
            #     start = timeit.default_timer()
            #     qrf_search = HalvingRandomSearchCV(qrf, param_grid, resource="n_estimators", max_resources=120, min_resources=10, scoring=cur_scorer, n_jobs=14, random_state=0)
            #     with parallel_backend('multiprocessing'):
            #         qrf_search.fit(list_vstack(x_train), list_vstack(y_train).squeeze())
            #     stop = timeit.default_timer()
            #     qrf_time['train'].append(stop - start)
            #     pprint(qrf_search.best_params_)
            #     QRF_model_dict['{0}'.format(each_q)] = qrf_search
            # # ------ QRF model predict
            # for cur_data, each_q in zip(test_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_test_nl, x_test_l, y_test, t_test = cur_data
            #     x_test = [np.hstack((each_x_nl, each_x_l, each_t)) for each_x_nl, each_x_l, each_t in zip(x_test_nl, x_test_l, t_test)]
            #     cur_model = QRF_model_dict['{0}'.format(each_q)]
            #     start = timeit.default_timer()
            #     cur_pred = cur_model.predict(list_vstack(x_test)).reshape((-1, 1))
            #     stop = timeit.default_timer()
            #     qrf_time['test'].append(stop - start)
            #     QRF_pred_list.append(cur_pred)
            # for cur_q in quantile_level:
            #     y_pred_qrf = np.quantile(np.array(QRF_pred_list).squeeze(), cur_q, axis=0).reshape((-1, 1))
            #     cur_year_QRF_qs_list.append(qs_score(list_vstack(y_test), y_pred_qrf, cur_q))
            # QRF_qs_list.append(cur_year_QRF_qs_list)

            # # ---- QGB
            # cur_year_QGB_qs_list = []
            # y_pred_med_QGB = []
            # param_grid = {"learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 5, 8], "min_samples_leaf": [1, 20, 40], "min_samples_split": [2, 20, 40]}
            # QGB_model_dict = {}
            # QGB_pred_list = []
            # ------ QGB model fit
            # for cur_data, each_q in zip(train_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_train_nl, x_train_l, y_train, t_train = cur_data
            #     x_train = [np.hstack((each_x_nl, each_x_l, each_t))  for each_x_nl, each_x_l, each_t in zip(x_train_nl, x_train_l, t_train)]
            #     qgb = GradientBoostingRegressor(loss="quantile", alpha=each_q, random_state=0)
            #     cur_scorer = make_scorer(mean_pinball_loss, alpha=each_q, greater_is_better=False)
            #     print("begin fit {0}-QGB...".format(each_q))
            #     start = timeit.default_timer()
            #     qgb_search = HalvingRandomSearchCV(qgb, param_grid, resource="n_estimators", max_resources=120, min_resources=10, scoring=cur_scorer, n_jobs=14, random_state=0).fit(list_vstack(x_train), list_vstack(y_train).squeeze())
            #     stop = timeit.default_timer()
            #     qgb_time['train'].append(stop - start)
            #     pprint(qgb_search.best_params_)
            #     QGB_model_dict['{0}'.format(each_q)] = qgb_search
            # # ------ QGB model predict
            # for cur_data, each_q in zip(test_data_of_quantile_levels, quantile_level_grid_qplvc):
            #     x_test_nl, x_test_l, y_test, t_test = cur_data
            #     x_test = [np.hstack((each_x_nl, each_x_l, each_t)) for each_x_nl, each_x_l, each_t in zip(x_test_nl, x_test_l, t_test)]
            #     cur_model = QGB_model_dict['{0}'.format(each_q)]
            #     start = timeit.default_timer()
            #     cur_pred = cur_model.predict(list_vstack(x_test)).reshape((-1, 1))
            #     stop = timeit.default_timer()
            #     qgb_time['test'].append(stop - start)
            #     QGB_pred_list.append(cur_pred)
            # for cur_q in quantile_level:
            #     y_pred_qgb = np.quantile(np.array(QGB_pred_list).squeeze(), cur_q, axis=0).reshape((-1, 1))
            #     if cur_q == 0.5:
            #         y_pred_med_QGB.append(y_pred_qgb)
            #     cur_year_QGB_qs_list.append(qs_score(list_vstack(y_test), y_pred_qgb, cur_q))
            # QGB_qs_list.append(cur_year_QGB_qs_list)

            # # ---- print result：qs
            # print_result_qs(QPLVC_qs_list, QRF_qs_list, QGB_qs_list, energy_type)

            # ---- visualization: shap
            visualize_q = 0.9
            best_params = {'min_samples_split':20, 'min_samples_leaf':1, 'max_depth':2, 'learning_rate':0.2, 'n_estimators':90} # best params for current visualize quantile level, obtain from the result of QGB fit
            x_train_nl, x_train_l, y_train, t_train = train_data_of_quantile_levels[0]
            cur_mean_std = train_mean_std_of_quantile_levels[0]
            x_train = [np.hstack((each_x_nl, each_x_l, each_t)) for each_x_nl, each_x_l, each_t in zip(x_train_nl, x_train_l, t_train)]
            qgb_best = GradientBoostingRegressor(loss="quantile", alpha=visualize_q, random_state=0, **best_params).fit(list_vstack(x_train), list_vstack(y_train).squeeze())
            x_train = pd.DataFrame(data=list_vstack(x_train), columns=qauntile2predictors_dict[energy_type][visualize_q][0] + ['constant'] + qauntile2predictors_dict[energy_type][visualize_q][1] + [vary_cov])
            get_shap_plot(qgb_best, x_train, energy_type, visualize_q, ['Area', 'Stair2', 'hvac_jizhong'], vary_cov, cur_mean_std)


        # # -- Evaluation: ws and cp
        #     # ---- QPLVC
        #     cur_year_QPLVC_ws_list = []
        #     cur_year_QPLVC_cp_list = []
        #     # ------ QPLVC model predict
        #     for cur_alpha in alpha_level:
        #         cur_alpha_l = cur_alpha / 2
        #         cur_alpha_u = 1 - cur_alpha / 2
        #         y_pred_qplvc_l = np.quantile(np.array(qplvc_pred_list).squeeze(), cur_alpha_l, axis=0).reshape((-1, 1))
        #         y_pred_qplvc_u = np.quantile(np.array(qplvc_pred_list).squeeze(), cur_alpha_u, axis=0).reshape((-1, 1))
        #         cur_year_QPLVC_ws_list.append(winkler_score(list_vstack(y_test), y_pred_qplvc_l, y_pred_qplvc_u, cur_alpha))
        #         cur_year_QPLVC_cp_list.append(coverage_probability(list_vstack(y_test), y_pred_qplvc_l, y_pred_qplvc_u))
        #     QPLVC_ws_list.append(cur_year_QPLVC_ws_list)
        #     QPLVC_cp_list.append(cur_year_QPLVC_cp_list)
        #
        #     # ---- QRF
        #     cur_year_QRF_ws_list = []
        #     cur_year_QRF_cp_list = []
        #     # ------ QRF model predict
        #     for cur_alpha in alpha_level:
        #         cur_q_upper = 1 - cur_alpha / 2
        #         cur_q_lower = cur_alpha / 2
        #         y_pred_qrf_upper = np.quantile(np.array(QRF_pred_list).squeeze(), cur_q_upper, axis=0).reshape((-1, 1))
        #         y_pred_qrf_lower = np.quantile(np.array(QRF_pred_list).squeeze(), cur_q_lower, axis=0).reshape((-1, 1))
        #         cur_year_QRF_ws_list.append(winkler_score(list_vstack(y_test), y_pred_qrf_lower, y_pred_qrf_upper, cur_alpha))
        #         cur_year_QRF_cp_list.append(coverage_probability(list_vstack(y_test), y_pred_qrf_lower, y_pred_qrf_upper))
        #     QRF_ws_list.append(cur_year_QRF_ws_list)
        #     QRF_cp_list.append(cur_year_QRF_cp_list)
        #
        #     # ---- QGB
        #     cur_year_QGB_ws_list = []
        #     cur_year_QGB_cp_list = []
        #     # ------ QGB model predict
        #     for cur_alpha in alpha_level:
        #         cur_q_upper = 1 - cur_alpha / 2
        #         cur_q_lower = cur_alpha / 2
        #         y_pred_qgb_upper = np.quantile(np.array(QGB_pred_list).squeeze(), cur_q_upper, axis=0).reshape((-1, 1))
        #         y_pred_qgb_lower = np.quantile(np.array(QGB_pred_list).squeeze(), cur_q_lower, axis=0).reshape((-1, 1))
        #         cur_year_QGB_ws_list.append(winkler_score(list_vstack(y_test), y_pred_qgb_lower, y_pred_qgb_upper, cur_alpha))
        #         cur_year_QGB_cp_list.append(coverage_probability(list_vstack(y_test), y_pred_qgb_lower, y_pred_qgb_upper))
        #     QGB_ws_list.append(cur_year_QGB_ws_list)
        #     QGB_cp_list.append(cur_year_QGB_cp_list)
        #
        #     # ---- print result： ws, cp and running time
        #     print_result_ws_cp(QPLVC_ws_list, QPLVC_cp_list, QRF_ws_list, QRF_cp_list, QGB_ws_list, QGB_cp_list, energy_type)
        #     if energy_type == 'Q':
        #         print_runtime(qplvc_time, qrf_time, qgb_time)





