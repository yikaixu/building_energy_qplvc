import pandas as pd
import numpy as np
import random as rd
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
from tqdm import tqdm
from change_point_model import cpm_fit, cpm_predict
from Bspline_model import Bspline_fit, Bspline_predict
from QPLVC_model import QPLVC_fit, QPLVC_predict
from QVC_model import QVC_fit, QVC_predict
from QNIS import qnis
import seaborn as sns
from scipy.interpolate import BSpline
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


os_sep = os.sep
data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep
raw_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "data"
meter_dict = ["electricity (J/m2)", "chilledwater (J/m2)", "steam (J/m2)", "hotwater (J/m2)"]


def id_trans(id):
    type, site, bid = id.split('_')[0], id.split('_')[1], id.split('_')[2]
    if site in [str(each_str) for each_str in range(10)]:
        site = '0' + site
    if bid in [str(each_str) for each_str in range(10)]:
        bid = '0' + bid
    return '_'.join([type, site, bid])


def data_load(meter_id, train_or_test):
    all_data = os.listdir(data_path + train_or_test)
    data = []
    building_id = []
    for each_data in all_data:
        if each_data.split(".")[0][-1] == str(meter_id):
            data.append(pd.read_csv(data_path + train_or_test + os_sep + each_data))
            cur_site_id = each_data.split('_')[1]
            cur_bid = each_data.split('_')[2]
            cur_type = each_data.split("_")[0]
            building_id.append("{0}_{1}_{2}".format(cur_type, cur_site_id, cur_bid))
    return data, building_id


def data_process(data, building_id, vary_cov, is_del_highmissing_col=True, is_contain_t=True, missing_rate=0.3):
    meter_str = meter_dict[meter_id]
    # delete features with high missing rate
    if is_del_highmissing_col:
        data_df = pd.concat(data)
        high_missing_features = data_df.loc[:, data_df.isnull().mean() > missing_rate].columns.tolist()
        for each_building in data:
           each_building.drop(high_missing_features, axis=1, inplace=True)
    # delete rows with missing
    proceseed_data = [each_data.dropna(axis=0, how='any') for each_data in data]
    # check the upper outlier threshold in response, hand selected
    upper_bar = np.quantile(list_vstack([item[meter_str].values.reshape((-1, 1)) for item in proceseed_data]), 0.99999)
    proceseed_data = [each_data[(each_data[meter_str] > 0) & (each_data[meter_str] < upper_bar)] for each_data in proceseed_data]
    t = []
    y = []
    x = []
    valid_building_id = []
    for each_data, each_bid in zip(proceseed_data, building_id):
        if len(each_data) != 0:
            y.append(each_data.loc[:, meter_str].values.reshape((-1, 1)))
            t.append(each_data.loc[:, vary_cov].values.reshape((-1, 1)))
            if is_contain_t:
                x.append(each_data.drop(labels=meter_str, axis=1).values.astype('float64'))
                feature_names = proceseed_data[0].columns.drop(labels=meter_str).tolist()
            else:
                x.append(each_data.drop(labels=[meter_str, vary_cov], axis=1).values.astype('float64'))
                feature_names = proceseed_data[0].columns.drop(labels=[meter_str, vary_cov]).tolist()
            valid_building_id.append(each_bid)
    return x, t, y, valid_building_id, feature_names


def building_meta_amplify(building_id, building_meta_data, t_list, df_output=False):
    building_meta_list = []
    for each_building, each_t in zip(building_id, t_list):
        cur_meta = building_meta_data[building_meta_data["id"] == id_trans(each_building)]
        building_meta_list.append(np.array([cur_meta.iloc[:, 1:].values] * len(each_t)).squeeze())
    if df_output:
        building_meta_list = [pd.DataFrame(data=item, columns=building_meta_data.columns[1:]) for item in building_meta_list]
        return building_meta_list
    else:
        return building_meta_list


# not used
def add_inter_term(x, building_meta, meta_predictors, weather_predictors):
    x_with_inter = x.copy()
    weather_predictors_with_inter = weather_predictors.copy()
    for meta_id, each_meta_predictor in enumerate(meta_predictors):
        for weather_id, each_weather_predictor in enumerate(weather_predictors):
            weather_predictors_with_inter.append("{0} * {1}".format(each_weather_predictor, each_meta_predictor))
            count = 0
            for each_x, each_meta, each_x_inter in zip(x, building_meta, x_with_inter):
                cur_wheather_data = each_x[:, weather_id].reshape((-1, 1))
                cur_meta_data = each_meta[:, meta_id].reshape((-1, 1))
                x_with_inter[count] = np.hstack((each_x_inter, cur_wheather_data * cur_meta_data))
                count += 1
    return x_with_inter, weather_predictors_with_inter


def reprocess(data_nl, screen_nl_name, vary_cov, data_l, screen_l_name, building_id):
    data_l_apl = building_meta_amplify(building_id, data_l, data_nl, df_output=True)
    data_all = [pd.concat([each_weather_data, each_meta_data], axis=1) for each_weather_data, each_meta_data in zip(data_nl, data_l_apl)]
    meter_str = meter_dict[meter_id]
    # ---- select features
    is_contain_t = vary_cov in screen_nl_name
    if is_contain_t:
        data_nl_l = [each.loc[:, [meter_str] + screen_nl_name + screen_l_name] for each in data_all]
    else:
        data_nl_l = [each.loc[:, [meter_str] + [vary_cov] + screen_nl_name + screen_l_name] for each in data_all]
    # ---- delete row contain missingness, response outlier and separate into x, t, y
    x_nl_l, t, y, valid_building_id, nl_l_names = data_process(data_nl_l, building_id, vary_cov, False, is_contain_t)
    # ---- select features
    x_l = [each[:, 3:6] for each in x_nl_l]
    x_nl = [each[:, 0:3] for each in x_nl_l]
    return t, y, x_nl, x_l, valid_building_id, nl_l_names


def list_vstack(list):
    result = list[0]
    for item in list[1:]:
        result = np.vstack((result, item))
    return result


def time_process(data):
    result = []
    timestamp_length = np.array([len(each_data) for each_data in data])
    assert np.max(timestamp_length) == 366 * 24
    max_id = np.argmax(timestamp_length)
    ref = pd.DataFrame(data=np.hstack((data[max_id]['timestamp'].values.reshape((-1, 1)), np.arange(366 * 24).reshape((-1, 1)))), columns=['timestamp', 'time_id'])
    for each_data in data:
        # each_data.drop(each_data[each_data['Time'] == 'NaT'].index, inplace=True)
        result.append(pd.merge(each_data, ref, on="timestamp").drop(labels='timestamp', axis=1))
    return result


def qs_score(y_true, y_pred, quantile_level):
    result = np.abs(y_true - y_pred)
    result[y_true < y_pred] = result[y_true < y_pred] * (1 - quantile_level)
    result[y_true >= y_pred] = result[y_true >= y_pred] * quantile_level
    return np.mean(result)


def normalize(data_list):
    data_all = list_vstack(data_list)
    mean_all = np.mean(data_all, axis=0).reshape((1, -1))
    std_all = np.std(data_all, axis=0).reshape((1, -1))
    return [(item - mean_all) / std_all for item in data_list]


if __name__ == "__main__":
    np.random.seed(20230419)
    k = 3
    dn = 11
    width = 0.5
    constant_scale_factor = 1
    vary_cov = 'air_temperature'

    for meter_id in [0, 1, 3]:
        # -- data process
        # ---- load meta data
        building_meta_train_raw = pd.read_csv(raw_data_path + os_sep + "train" + os_sep + "sample_20231020.csv")
        building_meta_test_raw = pd.read_csv(raw_data_path + os_sep + "test" + os_sep + "sample_20231020.csv")
        building_meta_features = building_meta_train_raw.columns.tolist()[1:]
        # ---- load whether data
        data_train, building_id_train_raw = data_load(meter_id, "processed_data_train_simulation")
        data_test, building_id_test_raw = data_load(meter_id, "processed_data_test_simulation")
        # ---- transform datetime to integer index
        data_train = time_process(data_train)
        data_test = time_process(data_test)
        # ---- process missing and outlier then separate into x, t, y
        x_train, t_train, y_train, building_id_train, weather_features = data_process(data_train, building_id_train_raw, vary_cov)
        _, t_test, y_test, building_id_test, _ = data_process(data_test, building_id_test_raw, vary_cov)
        # ---- amplify meta data
        building_meta_train = building_meta_amplify(building_id_train, building_meta_train_raw, t_train)
        # ---- add intersect term
        # x_train, weather_features = add_inter_term(x_train, building_meta_train, building_meta_features, weather_features)

        # -- separate data according building type
        y_train_LargeOffice = []; x_train_LargeOffice = []; meta_train_LargeOffice = []; t_train_LargeOffice = []
        y_train_LargeHotel = [];  x_train_LargeHotel = [];  meta_train_LargeHotel = []; t_train_LargeHotel = []
        for each_t, each_y, each_x, each_meta, each_bid in zip(t_train, y_train, x_train, building_meta_train, building_id_train):
            cur_type = each_bid.split('_')[0]
            if cur_type == 'LargeOffice':
                y_train_LargeOffice.append(each_y)
                t_train_LargeOffice.append(each_t)
                x_train_LargeOffice.append(each_x)
                meta_train_LargeOffice.append(each_meta)
            else:
                y_train_LargeHotel.append(each_y)
                t_train_LargeHotel.append(each_t)
                x_train_LargeHotel.append(each_x)
                meta_train_LargeHotel.append(each_meta)

        # # -- visualization: show density of y
        # fig, ax = plt.subplots()
        # y_train_LargeOffice_total = list_vstack(y_train_LargeOffice)
        # y_train_LargeHotel_total = list_vstack(y_train_LargeHotel)
        # sns.distplot(y_train_LargeOffice_total, hist=True, kde=True, color='red', hist_kws={'edgecolor': 'red'}, kde_kws={'linewidth': 1.5}, label='large office', bins=30)
        # sns.distplot(y_train_LargeHotel_total, hist=True, kde=True, color='blue', hist_kws={'edgecolor': 'blue'}, kde_kws={'linewidth': 1.5}, label='large hotel', bins=30)
        # plt.axvline(x=np.quantile(y_train_LargeOffice_total, 0.5), linestyle='--', linewidth=1.5, color='red')
        # plt.axvline(x=np.quantile(y_train_LargeHotel_total, 0.5), linestyle='--', linewidth=1.5, color='blue')
        # plt.xlabel('{0}(j/m\u00b2)'.format(meter_dict[meter_id].split(' ')[0]))
        # plt.xlim((0, 6e5))
        # plt.ylim((0, 0.85 * 1e-4))
        # plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        # plt.legend()
        # fig.savefig("y_train_density_meter{0}.pdf".format(meter_id))

        # # -- non-linear part feature screen
        # for tau in [0.1, 0.5, 0.9]:
        #     for cur_type in ['LargeOffice', 'LargeHotel']:
        #         print("begin non-linear part feature screen...")
        #         exec('cur_x = x_train_{0}'.format(cur_type))
        #         exec('cur_meta = meta_train_{0}'.format(cur_type))
        #         exec('cur_y = y_train_{0}'.format(cur_type))
        #         exec('cur_t = t_train_{0}'.format(cur_type))
        #         total_x_nl = np.hstack((list_vstack(cur_x), list_vstack(cur_meta)))
        #         total_y = list_vstack(cur_y)
        #         total_t = list_vstack(cur_t)
        #         x_nl_qnis_value = np.array(qnis(total_y, total_x_nl, total_t, tau, width, k, dn))
        #         screen_nl_df = pd.DataFrame(data=np.array([weather_features + building_meta_features, x_nl_qnis_value]).T, columns=['var', 'qnis_value_meter{0}'.format(meter_id)])
        #         np.random.shuffle(total_x_nl)
        #         x_nl_qnis_value_random = np.array(qnis(total_y, total_x_nl, total_t, tau, width, k, dn))
        #         cur_max_threshold = int(np.array(x_nl_qnis_value_random).max())
        #         cur_min_threshold = int(np.array(x_nl_qnis_value_random).min())
        #         screen_nl_df.to_csv("screen_nl_meter{0}_{1}_{2}_min{3}_max{4}.csv".format(meter_id, cur_type, tau, cur_min_threshold, cur_max_threshold), index=False)

        # -- selected features
        # ---- large office
        x_nl_screen_names_LargeOffice_meter0_lower = ['EEPD', 'LPD', 'infiltration']
        x_nl_screen_names_LargeOffice_meter1_lower = ['CSPT', 'SHGC', 'solarabs_roof']
        x_nl_screen_names_LargeOffice_meter3_lower = ['infiltration', 'HSPT', 'time_id']

        x_l_screen_names_LargeOffice_meter0_lower = ['SHGC', 'solarabs_roof', 'CSPT']
        x_l_screen_names_LargeOffice_meter1_lower = ['EEPD', 'AFPP', 'LPD']
        x_l_screen_names_LargeOffice_meter3_lower = ['wind_speed', 'solarabs_wall', 'AFPP']

        x_nl_screen_names_LargeOffice_meter0_middle = ['EEPD', 'COP', 'LPD']
        x_nl_screen_names_LargeOffice_meter1_middle = ['CSPT', 'dew_temperature', 'SHGC']
        x_nl_screen_names_LargeOffice_meter3_middle = ['AFPP', 'HSPT', 'SHGC']

        x_l_screen_names_LargeOffice_meter0_middle = ['dew_temperature', 'wind_speed', 'CSPT']
        x_l_screen_names_LargeOffice_meter1_middle = ['EEPD', 'LPD', 'AFPP']
        x_l_screen_names_LargeOffice_meter3_middle = ['solarabs_roof', 'wind_speed', 'FAPP']

        x_nl_screen_names_LargeOffice_meter0_upper = ['EEPD', 'LPD', 'COP']
        x_nl_screen_names_LargeOffice_meter1_upper = ['FAPP', 'dew_temperature', 'AFPP']
        x_nl_screen_names_LargeOffice_meter3_upper = ['AFPP', 'FAPP', 'EEPD']

        x_l_screen_names_LargeOffice_meter0_upper = ['CSPT', 'time_id', 'solarabs_roof']
        x_l_screen_names_LargeOffice_meter1_upper = ['CSPT', 'FTE', 'LPD']
        x_l_screen_names_LargeOffice_meter3_upper = ['solarabs_roof', 'HSPT', 'LPD']

        # ---- large hotel
        x_nl_screen_names_LargeHotel_meter0_lower = ['COP', 'FAPP', 'SHGC']
        x_nl_screen_names_LargeHotel_meter1_lower = ['FAPP', 'dew_temperature', 'FTE']
        x_nl_screen_names_LargeHotel_meter3_lower = ['time_id', 'HSPT', 'SHGC']

        x_l_screen_names_LargeHotel_meter0_lower = ['LPD', 'EEPD', 'FTE']
        x_l_screen_names_LargeHotel_meter1_lower = ['sea_level_pressure', 'CSPT', 'HSPT']
        x_l_screen_names_LargeHotel_meter3_lower = ['AFPP', 'COP', 'FTE']

        x_nl_screen_names_LargeHotel_meter0_middle = ['COP', 'LPD', 'FAPP']
        x_nl_screen_names_LargeHotel_meter1_middle = ['dew_temperature', 'CSPT', 'FAPP']
        x_nl_screen_names_LargeHotel_meter3_middle = ['FAPP', 'AFPP', 'HSPT']

        x_l_screen_names_LargeHotel_meter0_middle = ['SHGC', 'CSPT', 'EEPD']
        x_l_screen_names_LargeHotel_meter1_middle = ['HSPT', 'LPD', 'FTE']
        x_l_screen_names_LargeHotel_meter3_middle = ['EEPD', 'CSPT', 'COP']

        x_nl_screen_names_LargeHotel_meter0_upper = ['COP', 'LPD', 'AFPP']
        x_nl_screen_names_LargeHotel_meter1_upper = ['FAPP', 'HSPT', 'dew_temperature']
        x_nl_screen_names_LargeHotel_meter3_upper = ['FAPP', 'AFPP', 'solarabs_roof']

        x_l_screen_names_LargeHotel_meter0_upper = ['SHGC', 'FAPP', 'solarabs_roof']
        x_l_screen_names_LargeHotel_meter1_upper = ['EEPD', 'CSPT', 'LPD']
        x_l_screen_names_LargeHotel_meter3_upper = ['LPD', 'HSPT', 'COP']

        # -- fit, test and visualization
        plot_meter_id = [0, 1]
        plot_type = ['LargeHotel', 'LargeOffice']
        plot_quantile = 'middle' # could be any of lower middle upper
        tau_dict = {'lower': 0.1, 'middle': 0.5, 'upper': 0.9}
        y_test_plot = []; t_test_plot = []
        for cur_type in ['LargeOffice', 'LargeHotel']:
            for cur_quantile_level in ['lower', 'middle', 'upper']:
                exec('cur_nl_features = x_nl_screen_names_{0}_meter{1}_{2}'.format(cur_type, meter_id, cur_quantile_level))
                exec('cur_l_features = x_l_screen_names_{0}_meter{1}_{2}'.format(cur_type, meter_id, cur_quantile_level))
                # ---- separate according type
                cur_data_train = []; cur_building_id_train = []
                for each_bid, each_building in zip(building_id_train_raw, data_train):
                    if cur_type == each_bid.split('_')[0]:
                        cur_data_train.append(each_building)
                        cur_building_id_train.append(each_bid)
                cur_data_test = []; cur_building_id_test = []
                for each_bid, each_building in zip(building_id_test_raw, data_test):
                    if cur_type == each_bid.split('_')[0]:
                        cur_data_test.append(each_building)
                        cur_building_id_test.append(each_bid)
                # ---- reprocess
                cur_t_train, cur_y_train, cur_x_train, cur_meta_train, _, _ = reprocess(cur_data_train, cur_nl_features, vary_cov, building_meta_train_raw, cur_l_features, cur_building_id_train)
                cur_x_train = normalize(cur_x_train)
                cur_meta_train = normalize(cur_meta_train)
                cur_x_train = [np.hstack((each, constant_scale_factor * np.ones(len(each)).reshape((-1, 1)))) for each in cur_x_train]
                cur_t_test, cur_y_test, cur_x_test, cur_meta_test, _, _ = reprocess(cur_data_test, cur_nl_features, vary_cov, building_meta_test_raw, cur_l_features, cur_building_id_test)
                cur_x_test = normalize(cur_x_test)
                cur_meta_test = normalize(cur_meta_test)
                cur_x_test = [np.hstack((each, constant_scale_factor * np.ones(len(each)).reshape((-1, 1)))) for each in cur_x_test]
                cur_nl_features.append("constant")
                # ---- delete records t not in the range of training set
                t_train_max = np.array([np.max(item) for item in cur_t_train]).max()
                t_train_min = np.array([np.min(item) for item in cur_t_train]).min()
                remove_id = [((each >= t_train_min) & (each <= t_train_max)).squeeze() for each in cur_t_test]
                exec("cur_t_test_{0} = [each[remove_id[id]] for id, each in enumerate(cur_t_test)]".format(cur_quantile_level))
                exec("cur_y_test_{0} = [each[remove_id[id]] for id, each in enumerate(cur_y_test)]".format(cur_quantile_level))
                cur_meta_test = [each[remove_id[id]] for id, each in enumerate(cur_meta_test)]
                cur_x_test = [each[remove_id[id]] for id, each in enumerate(cur_x_test)]
                dn_list = cur_x_train[0].shape[1] * [dn]
                cur_tau = tau_dict[cur_quantile_level]
                # # ---- fit quantile model
                # print("begin fit {0}-QPLVC...".format(cur_tau))
                # cur_beta, cur_knots = QPLVC_fit(cur_x_train, cur_meta_train, cur_t_train, cur_y_train, dn_list, k, cur_tau)
                # exec("cur_y_pred_{0}, _ = QPLVC_predict(cur_x_test, cur_meta_test, cur_t_test_{0}, k, cur_beta, cur_knots)".format(cur_quantile_level))
                # # ---- visualization: dynamic coefficient
                # if (meter_id in plot_meter_id) and (cur_type in plot_type):
                #     fig, ax = plt.subplots()
                #     t_plot = np.linspace(-20, 40, 1000)
                #     for fid, each_feature, each_color, each_line_style in zip(range(len(cur_nl_features)), cur_nl_features, ["red", "blue", "green", "black"], ["--", "-", "-.", ":"]):
                #         knots_plot = cur_knots[fid]
                #         beta_plot = cur_beta[dn * fid: dn * (fid + 1)]
                #         cur_B_mat_plot = BSpline.design_matrix(t_plot, knots_plot, k).toarray()
                #         cur_coef_plot = np.dot(cur_B_mat_plot, beta_plot)
                #         plt.plot(t_plot, cur_coef_plot, linestyle=each_line_style, label=each_feature.replace("_", " "), color=each_color)
                #     plt.legend(loc="upper left")
                #     plt.grid()
                #     plt.ylabel("coefficient value")
                #     plt.xlabel(vary_cov.replace("_", " "))
                #     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                #     fig.savefig("coef_meter{0}_{1}_{2}.pdf".format(meter_id, cur_type, cur_quantile_level))
                # if cur_quantile_level == plot_quantile:
                #     y_test_plot.append(cur_y_test)
                #     t_test_plot.append(cur_t_test)

                # ---- ablation study
                beta_cpm, cp_list, cpm_type = cpm_fit(cur_t_train, cur_y_train, t_train_min, t_train_max, 1, cur_tau)
                print("current change point model type is {0}".format(cpm_type))
                exec("y_pred_cpm = cpm_predict(cur_t_test_{0}, beta_cpm, cp_list, cpm_type)".format(cur_quantile_level))
                beta_bspline, knots_bspline = Bspline_fit(cur_t_train, cur_y_train, k, dn, cur_tau)
                exec("y_pred_bspline = Bspline_predict(cur_t_test_{0}, k, beta_bspline, knots_bspline)".format(cur_quantile_level))
                beta_qplvc, knots_qplvc = QPLVC_fit(cur_x_train, cur_meta_train, cur_t_train, cur_y_train, dn_list, k, cur_tau)
                exec("y_pred_qplvc, _ = QPLVC_predict(cur_x_test, cur_meta_test, cur_t_test_{0}, k, beta_qplvc, knots_qplvc)".format(cur_quantile_level))
                cur_x_meta_train = [np.hstack((each_x, each_meta)) for each_x, each_meta in zip(cur_x_train, cur_meta_train)]
                beta_qvc, knots_qvc = QVC_fit(cur_x_meta_train, cur_t_train, cur_y_train, cur_x_meta_train[0].shape[1] * [dn], k, cur_tau)
                cur_x_meta_test = [np.hstack((each_x, each_meta)) for each_x, each_meta in zip(cur_x_test, cur_meta_test)]
                exec("y_pred_qvc, _ = QVC_predict(cur_x_meta_test, cur_t_test_{0}, k, beta_qvc, knots_qvc)".format(cur_quantile_level))
                # ---- print result
                print(" ***** {0} quantile, meter {1}, type:{2} *****".format(cur_tau, meter_id, cur_type))
                exec("cur_y_test = cur_y_test_{0}".format(cur_quantile_level))
                print("change point:{0}  bspline:{1}  qplvc:{2}  qvc:{3}".format(qs_score(list_vstack(cur_y_test), y_pred_cpm, cur_tau),
                                                                                 qs_score(list_vstack(cur_y_test), y_pred_bspline, cur_tau),
                                                                                 qs_score(list_vstack(cur_y_test), y_pred_qplvc, cur_tau),
                                                                                 qs_score(list_vstack(cur_y_test), y_pred_qvc, cur_tau)))

        #     # ---- visualization: prediction visualization
        #     ylim_list = [4 * 1e5, 5 * 1e5, 0, 2 * 1e5]
        #     fig, ax = plt.subplots()
        #     plt.scatter(list_vstack(cur_t_test_upper), cur_y_pred_upper, marker='o', color='blue', s=0.5, alpha=0.02)
        #     plt.scatter(list_vstack(cur_t_test_middle), cur_y_pred_middle, marker='o', color='green', s=0.5, alpha=0.02)
        #     plt.scatter(list_vstack(cur_t_test_lower), cur_y_pred_lower, marker='o', color='gray', s=0.5, alpha=0.02)
        #     blue_patch = mpatches.Patch(color='blue', label='predict:0.9')
        #     green_patch = mpatches.Patch(color='green', label='predict:0.5')
        #     gray_patch = mpatches.Patch(color='gray', label='predict:0.1')
        #     ax.legend(handles=[blue_patch, green_patch, gray_patch], loc='upper right')
        #     plt.xlabel('air temperature')
        #     plt.ylabel('j/m\u00b2')
        #     plt.ylim((0, ylim_list[meter_id]))
        #     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        #     fig.savefig("meter{0}_{1}_test.png".format(meter_id, cur_type))
        #
        # # ---- visualization: show building type differences of test y
        # fig, ax = plt.subplots()
        # for each_y, each_t in zip(y_test_plot[0], t_test_plot[0]):
        #     plt.scatter(each_t, each_y, marker='o', color='red', s=0.5, label='large office', alpha=0.03)
        # for each_y, each_t in zip(y_test_plot[1], t_test_plot[1]):
        #     plt.scatter(each_t, each_y, marker='o', color='blue', s=0.5, label='large hotel', alpha=0.03)
        # red_patch = mpatches.Patch(color='red', label='large office')
        # blue_patch = mpatches.Patch(color='blue', label='large hotel')
        # ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        # plt.xlabel(vary_cov.replace("_", " "))
        # plt.ylabel('j/m\u00b2')
        # plt.ylim((0, ylim_list[meter_id]))
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # fig.savefig("y_test_scatter_meter{0}.png".format(meter_id))









