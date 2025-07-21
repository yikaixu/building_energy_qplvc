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


# print data summary information
raw_building_data = pd.read_excel(raw_data_path_prefix + os_sep + "5 train_building_info.xlsx")[['Stair1', 'Stair2', 'Area']]
print("building_min")
print(raw_building_data.min())
print("building_max")
print(raw_building_data.max())
print("building_median")
print(raw_building_data.median())


train_weather = pd.concat([raw_data_weather_2015, raw_data_weather_2016], axis=0, ignore_index=True)
print('min_train')
print(train_weather.min())
print('max_train')
print(train_weather.max())
print('median_train')
print(train_weather.median())
print('min_test')
print(raw_data_weather_2017.min())
print('max_test')
print(raw_data_weather_2017.max())
print('median_test')
print(raw_data_weather_2017.median())

energy_all = pd.read_csv(raw_data_path_prefix + os_sep + "6 train.csv")
energy_all['Time'] = pd.to_datetime(energy_all['Time'])
print('min_train_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'Q')]['Record'].min())
print('max_train_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'Q')]['Record'].max())
print('median_train_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'Q')]['Record'].median())
print('min_train_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'W')]['Record'].min())
print('max_train_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'W')]['Record'].max())
print('median_train_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2015, 2016])) & (energy_all['Type'] == 'W')]['Record'].median())

print('min_test_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'Q')]['Record'].min())
print('max_test_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'Q')]['Record'].max())
print('median_test_Q')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'Q')]['Record'].median())
print('min_test_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'W')]['Record'].min())
print('max_test_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'W')]['Record'].max())
print('median_test_W')
print(energy_all[(energy_all['Time'].dt.year.isin([2017])) & (energy_all['Type'] == 'W')]['Record'].median())


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
