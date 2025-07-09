import pandas as pd
import os


os_sep = os.sep
save_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "processed_realdata"


raw_data_path_prefix =  os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os_sep + "data" + os_sep + "real_data_raw"
raw_data_weather_2015 = pd.read_excel(raw_data_path_prefix + os_sep + "4 weather" + os_sep + "shanghai weather_2015.xlsx").rename(columns = {'time':'Time', '温度（℃）':'air temperature', '露点温度（℃）':'dew temperature', '相对湿度（%）':'relative humidity', '气压（pa）':'pressure', '风速（m/s）':'wind speed'})
raw_data_weather_2016 = pd.read_excel(raw_data_path_prefix + os_sep + "4 weather" + os_sep + "shanghai weather_2016.xlsx").rename(columns = {'time':'Time', '温度（℃）':'air temperature', '露点温度（℃）':'dew temperature', '相对湿度（%）':'relative humidity', '气压（pa）':'pressure', '风速（m/s）':'wind speed'})
raw_data_weather_2017 = pd.read_excel(raw_data_path_prefix + os_sep + "4 weather" + os_sep + "shanghai weather_2017.xlsx").rename(columns = {'time':'Time', '温度（℃）':'air temperature', '露点温度（℃）':'dew temperature', '相对湿度（%）':'relative humidity', '气压（pa）':'pressure', '风速（m/s）':'wind speed'})
raw_data_energy_total = pd.read_csv(raw_data_path_prefix + os_sep + "6 train.csv")


def date_to_string(df):
    df['Time'] = df['Time'].astype(str)
    return df


for item in list(raw_data_energy_total.groupby('Type')):
    cur_energy_type = item[0]
    cur_energy_type_data = item[1]
    for sub_item in list(cur_energy_type_data.groupby('BuildingID')):
        cur_bid = sub_item[0]
        cur_bid_data = sub_item[1]
        cur_merge_data_2015 = pd.merge(date_to_string(raw_data_weather_2015), date_to_string(cur_bid_data), on = "Time", how = 'left').drop(['Type', 'BuildingID'], axis = 1)
        cur_merge_data_2016 = pd.merge(date_to_string(raw_data_weather_2016), date_to_string(cur_bid_data), on = "Time", how = 'left').drop(['Type', 'BuildingID'], axis = 1)
        cur_merge_data_2017 = pd.merge(date_to_string(raw_data_weather_2017), date_to_string(cur_bid_data), on = "Time",  how = 'left').drop(['Type', 'BuildingID'], axis = 1)
        cur_merge_data_2015.to_csv(save_path + os_sep + '{0}_{1}_{2}.csv'.format(cur_bid, cur_energy_type, 2015), index = False)
        cur_merge_data_2016.to_csv(save_path + os_sep + '{0}_{1}_{2}.csv'.format(cur_bid, cur_energy_type, 2016), index = False)
        cur_merge_data_2017.to_csv(save_path + os_sep + '{0}_{1}_{2}.csv'.format(cur_bid, cur_energy_type, 2017), index = False)
