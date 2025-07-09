import numpy as np
import pandas as pd
import os

for current_data_type in ["train", "test"]:
    os_sep = os.sep
    save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "processed_data_{0}_simulation".format(current_data_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    raw_simulation_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "data" + os_sep + current_data_type
    raw_real_data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "data" + os_sep + "ashrae-energy-prediction"

    meter_dict = {"electricity (J/m2)":0, "chilledwater (J/m2)":1, "steam (J/m2)":2, "hotwater (J/m2)":3}

    weather_all_site = pd.read_csv(raw_real_data_path + os_sep + "weather_train.csv")
    all_buildings = os.listdir(raw_simulation_data_path + os_sep + "result_20231020")
    for cur_building_name in all_buildings:
        cur_building = pd.read_csv(raw_simulation_data_path + os_sep + "result_20231020" + os_sep + cur_building_name)
        cur_site_id = int(cur_building_name.split("_")[1])
        cur_type = cur_building_name.split("_")[0]
        cur_bid = int(cur_building_name.split("_")[2].split(".")[0])
        for meter_id in ["electricity (J/m2)", "chilledwater (J/m2)", "steam (J/m2)", "hotwater (J/m2)"]:
            cur_output_file_name = "{0}_{1}_{2}_{3}.csv".format(cur_type, cur_site_id, cur_bid, meter_dict[meter_id])
            cur_output_left = cur_building.loc[:, ["timestamp", meter_id]]
            cur_output_right = weather_all_site[weather_all_site['site_id'] == cur_site_id].drop(["site_id"], axis = 1)
            cur_output_file = pd.merge(cur_output_left, cur_output_right, on = "timestamp")
            cur_output_file.to_csv(save_path + os_sep + cur_output_file_name, index=False)

