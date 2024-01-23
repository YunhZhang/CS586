import os
import pandas as pd
import numpy as np
import csv
import gc
import sys
import statistics
from glob import glob
from tqdm import trange
import pickle


class DataProcess:
    def __init__(self, data, grid=6):
        self.trajectory = pd.DataFrame(data, columns=['plate', 'longitude', 'latitude', 'time', 'status'])
        self.grid = grid
        longi_list = self.trajectory['longitude'].values
        lati_list = self.trajectory['latitude'].values
        self.longi_max = longi_list.max()
        self.longi_min = longi_list.min()
        self.lati_max = lati_list.max()
        self.lati_min = lati_list.min()
    
    @staticmethod
    def time_to_second(time_series):
        time = pd.Timestamp(time_series)
        return time.hour * 60 * 60 + time.minute * 60 + time.second
    
    @staticmethod
    def time_to_min(time_series):
        time = pd.Timestamp(time_series)
        return time.minute
    
    @staticmethod
    def time_to_hour(time_series):
        time = pd.Timestamp(time_series)
        return time.hour
    
    @staticmethod
    def time_to_day(time_series):
        time = pd.Timestamp(time_series)
        return time.day

    @staticmethod
    def time_to_month(time_series):
        time = pd.Timestamp(time_series)
        return time.month

    @staticmethod
    def time_to_dayofweek(time_series):
        time = pd.Timestamp(time_series)
        return time.dayofweek
    
    def coordinate_to_id(self, traj):
        long_range = [113.5, 114.7]
        longi_val = traj['longitude']
        lati_val = traj['latitude']
        longi_grid = np.linspace(long_range[0], long_range[1], self.grid)
        lati_grid = np.linspace(22, 23, self.grid)
        long_idx = (longi_grid <= longi_val).sum()
        lat_idx = (lati_grid <= lati_val).sum()
        id = (lat_idx - 1) * self.grid + (long_idx - 1)
        return id 
    
    def data_process(self, traj_raw, traj):
        traj['longitude'] = traj_raw['longitude'].values
        traj['latitude'] = traj_raw['latitude'].values
        traj['status'] = traj_raw['status']
        time = traj_raw['time']
        traj['seconds'] = time.apply(self.time_to_second).values
        traj['minutes'] = time.apply(self.time_to_min).values
        traj['hour'] = time.apply(self.time_to_hour).values
        traj['date'] = time.apply(self.time_to_day).values
        traj['month'] = time.apply(self.time_to_month).values
        traj['day'] = time.apply(self.time_to_dayofweek).values
        traj['grid_id'] = traj_raw.apply(self.coordinate_to_id, axis=1).values

        if len(traj_raw['plate'].unique()) == 0:
            traj['plate'] = []
            traj['most_freq_grid'] = []
            traj['service_time'] = []
            traj['status_mean'] = []
        else:
            traj['plate'] = traj_raw['plate'].unique()[0]
            traj['most_freq_grid'] = statistics.mode(traj['grid_id'])
            traj['service_time'] = traj['seconds'].max() - traj['seconds'].min()
            traj['status_mean'] = traj_raw['status'].values.mean()
        return traj

    def seperate_data(self):
        data = self.trajectory.sort_values(by=['plate', 'status', 'time'])
        processed_data = {}
        num_drivers = len(data['plate'].unique())      
        for i in range(num_drivers):
            driver = f'driver_{str(i+1)}'
            data_temp = data.loc[data['plate'] == i]
            processed_data[driver] = {}
            for i_status in range(2):
                status = f'status_{str(i_status)}'
                processed_data[driver][status] = {}
                # try:
                #     assert len(data_temp) > 0, 'zero-size data frame!'
                # except AssertionError:
                #     print('driver %s is empty' % (i+1))
                #     continue
                data_temp_stat = data_temp.loc[data_temp['status'] == i_status]
                processed_data[driver][status] = self.data_process(data_temp_stat, processed_data[driver][status])
                gc.collect()
        return processed_data
    

    
if __name__ == '__main__':
    outside_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = 'data_5drivers'
    data_path = os.path.join(outside_path, folder_name)
    csv_files = os.listdir(data_path)
    csv_files = [os.path.join(data_path, c) for c in csv_files if c.endswith(".csv")]

    save_folder = 'training data separate'
    save_path = os.path.join(outside_path,save_folder)
    os.makedirs(save_path,exist_ok=True)

    data_list = []
    for i in trange(len(csv_files)):
        file_path = csv_files[i]
        file_name = file_path.split('/')[-1].split('.')[0]
        traj = pd.read_csv(file_path)
        try:
            assert len(traj) > 0, 'Empty data set!'
        except AssertionError:
            print('problem file: %s' % file_path)
            continue
        data = DataProcess(traj)
        new_traj = data.seperate_data()

        for i in range(5):
            driver = f'driver_{i+1}'
            save_file_name = f'{driver}_{file_name}.pkl'
            save_file_path = os.path.join(save_path, save_file_name)
            data_save = new_traj[driver]
            if len(data_save) == 0:
                print('{} on {} is empty'.format(driver, file_name))
                # continue
            with open(save_file_path, 'wb') as fp:
                pickle.dump(new_traj[driver], fp)
                # print('saving!')
        data_list.append(new_traj)