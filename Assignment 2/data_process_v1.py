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
        self.lati_max = longi_list.max()
        self.lati_min = longi_list.min()
    
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
        longi_val = traj['longitude']
        lati_val = traj['latitude']
        longi_grid = np.linspace(113.5, 114.7, self.grid)
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

    def run_data_new(self):
        data = self.trajectory.sort_values(by=['plate', 'status', 'time'])
        data_new = {}
        num_drivers = len(data['plate'].unique())
        assert num_drivers == 5, 'it should have 5 drivers, please check'
        for i_driver in range(num_drivers):
            drive_name = f'driver_{str(i_driver+1)}'
            data_temp = data.loc[data['plate'] == i_driver]
            data_new[drive_name] = {}
            try:
                assert len(data_temp) > 0, 'zero-size data frame!'
            except AssertionError:
                print('driver %s is empty' % (i_driver+1))
                continue
            # data_temp_stat = data_temp.loc[data_temp['status'] == i_status]
            data_new[drive_name] = self.data_process(data_temp, data_new[drive_name])
            gc.collect()
        return data_new
    

    
if __name__ == '__main__':
    outside_path = os.path.dirname(os.path.realpath(__file__))
    folder_name = 'data_5drivers'
    data_path = os.path.join(outside_path, folder_name)
    csv_files = os.listdir(data_path)
    csv_files = [os.path.join(data_path, c) for c in csv_files if c.endswith(".csv")]

    save_folder = 'training data'
    save_path = os.path.join(outside_path,save_folder)
    os.makedirs(save_path,exist_ok=True)

    data_list = []
    for i in trange(len(csv_files)):
        file_path = csv_files[i]
        file_name = file_path.split('/')[-1].split('.')[0]
        data_raw = pd.read_csv(file_path)
        try:
            assert len(data_raw) > 0, 'Empty data set!'
        except AssertionError:
            print('problem file: %s' % file_path)
            continue
        data_process = DataProcess(data_raw)
        data_new = data_process.run_data_new()

        for i_driver in range(5):
            driver_name = f'driver_{i_driver+1}'
            save_file_name = f'{driver_name}_{file_name}.pkl'
            save_file_path = os.path.join(save_path, save_file_name)
            data_save = data_new[driver_name]
            if len(data_save) == 0:
                print('{} on {} is empty'.format(driver_name, file_name))
                continue
            with open(save_file_path, 'wb') as fp:
                pickle.dump(data_new[driver_name], fp)
                # print('saving!')
        data_list.append(data_new)