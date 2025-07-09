import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from main_simulation import data_load, time_process, list_vstack, building_meta_amplify, normalize
from QVC_model import QVC_fit, QVC_predict


os_sep = os.sep
raw_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "data"
meter_dict = ["electricity (J/m2)", "chilledwater (J/m2)", "steam (J/m2)", "hotwater (J/m2)"]
tau_dict = {'lower': 0.1, 'middle': 0.5, 'upper': 0.9}
vary_cov = 'air_temperature'
dn = 11
k = 3


def data_process(data, building_id, vary_cov):
    meter_str = meter_dict[meter_id]
    # delete rows with missing
    proceseed_data = [each_data.dropna(axis=0, how='any') for each_data in data]
    # check the upper outlier threshold in response, hand selected
    upper_bar = np.quantile(list_vstack([item[meter_str].values.reshape((-1, 1)) for item in proceseed_data]), 0.99999)
    proceseed_data = [each_data[(each_data[meter_str] > 0) & (each_data[meter_str] < upper_bar)] for each_data in proceseed_data]
    t = []
    y = []
    valid_building_id = []
    for each_data, each_bid in zip(proceseed_data, building_id):
        if len(each_data) != 0:
            y.append(each_data.loc[:, meter_str].values.reshape((-1, 1)))
            t.append(each_data.loc[:, vary_cov].values.reshape((-1, 1)))
            valid_building_id.append(each_bid)
    return t, y, valid_building_id


def get_conditional_mean(t_list, y_list, t_plot, width=50):
    result = []
    t_np = t_list[0]
    y_np = y_list[0]
    for cur_t in t_plot:
        closet_id = np.argsort(np.abs(cur_t - t_np).squeeze())[0:width]
        result.append(y_np[closet_id].mean())
    return np.array(result)


def extract_current_type(given_type, data_list, building_id_list):
    result_data = []
    result_id = []
    for each_data, each_id in zip(data_list, building_id_list):
        each_type = each_id.split('_')[0]
        if given_type == each_type:
            result_data.append(each_data)
            result_id.append(each_id)
    return result_data, result_id


for meter_id in [0, 1]:
    for cur_type in ['LargeOffice']:
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
        # ---- extract t and y
        data_train = [each.loc[:, [meter_dict[meter_id], vary_cov]] for each in data_train]
        data_test = [each.loc[:, [meter_dict[meter_id], vary_cov]] for each in data_test]
        # ---- extract current type data
        data_train, building_id_train_raw = extract_current_type(cur_type, data_train, building_id_train_raw)
        data_test, building_id_test_raw = extract_current_type(cur_type, data_test, building_id_test_raw)
        # ---- process missing and outlier then separate into t, y
        t_train, y_train, building_id_train = data_process(data_train, building_id_train_raw, vary_cov)
        t_test, y_test, building_id_test = data_process(data_test, building_id_test_raw, vary_cov)
        # ---- extract test building data
        test_building_id = 0
        t_test = [t_test[test_building_id]]
        y_test = [y_test[test_building_id]]
        building_id_test = [building_id_test[test_building_id]]
        # ---- delete records t not in the range of training set
        t_train_max = np.array([np.max(item) for item in t_train]).max()
        t_train_min = np.array([np.min(item) for item in t_train]).min()
        remove_id = [((each >= t_train_min) & (each <= t_train_max)).squeeze() for each in t_test]
        t_test = [each[remove_id[id]] for id, each in enumerate(t_test)]
        y_test = [each[remove_id[id]] for id, each in enumerate(y_test)]
        # ---- amplify meta data
        building_meta_train = building_meta_amplify(building_id_train, building_meta_train_raw, t_train)
        # ---- normalize and add constant
        building_meta_train = normalize(building_meta_train)
        building_meta_train = [np.hstack((each, np.ones(len(each)).reshape((-1, 1)))) for each in building_meta_train]

        # -- fit qvc model
        beta_dict = {}
        knots_dict = {}
        for cur_quantile_level in ['lower', 'middle', 'upper']:
            cur_tau = tau_dict[cur_quantile_level]
            print(f'begin fit {cur_tau} qvc model ...')
            beta_qvc, knots_qvc = QVC_fit(building_meta_train, t_train, y_train, building_meta_train[0].shape[1] * [dn], k, cur_tau)
            beta_dict[cur_quantile_level] = beta_qvc
            knots_dict[cur_quantile_level] = knots_qvc

        # -- testing
        t_plot = np.linspace(t_test[0].min(), t_test[0].max(), 1000)
        t_plot_list = [t_plot.reshape((-1, 1))]
        # ---- amplify, normalize and add constant to test meta data
        building_meta_test = building_meta_amplify(building_id_test, building_meta_test_raw, t_plot_list)
        building_meta_test = normalize(building_meta_test)
        building_meta_test = [np.hstack((each, np.ones(len(each)).reshape((-1, 1)))) for each in building_meta_test]
        prediction_dict = {}
        for cur_quantile_level in ['lower', 'middle', 'upper']:
            cur_predict, _ = QVC_predict(building_meta_test, t_plot_list, k, beta_dict[cur_quantile_level], knots_dict[cur_quantile_level])
            prediction_dict[cur_quantile_level] = cur_predict

        # -- compute the real conditional response
        y_real = get_conditional_mean(t_test, y_test, t_plot)

        # -- plot
        # prediction_dict = {'lower': y_real - 20000, 'middle': y_real - 1, 'upper': y_real + 20000}
        fig, ax = plt.subplots()
        plt.fill_between(t_plot, prediction_dict['lower'].squeeze(), prediction_dict['upper'].squeeze(), color='gray', alpha=0.3)
        plt.plot(t_plot, prediction_dict['middle'].squeeze(), label='0.5 quantile curve', linestyle='--', color='black')
        plt.plot(t_plot, y_real, label='target building', linestyle='-', color='red')
        plt.legend(loc='upper left')
        plt.xlabel(vary_cov.replace("_", " "))
        plt.ylabel('j/m\u00b2')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        fig.savefig("benchmark_meter{0}_{1}_{2}.pdf".format(meter_id, building_id_test[0], vary_cov))

