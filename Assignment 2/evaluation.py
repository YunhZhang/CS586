import os
import sys
from glob import glob
from tqdm import trange
import pickle
import pandas as pd
import gc
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
from data_process_v2 import DataProcess
import NN_Module_v2 as NN_Module
from train_v2 import TrajLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def process_data(traj):
    data = DataProcess(traj)
    return data

def run(data_path, model):
    validation_data = TrajLoader(data_path)
    batch = 10
    params_v = {'batch_size':batch,'shuffle':False,'num_workers':0}
    data_val = torch.utils.data.DataLoader(validation_data, **params_v)
    
    prediction = []
    for i, traj_test in enumerate(data_val):
        traj_0 = traj_test['data_s0'].to(device)
        traj_1 = traj_test['data_s1'].to(device)
        label_gt = traj_test['labels'].to(device)
        
        output_val = model(traj_0,traj_1)
        prediction.append(output_val)
    
    return prediction

if __name__ == '__main__':
#### Load the data
    outside_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_name = 'data'
    data_path = os.path.join(outside_path, data_folder_name)
    csv_files = os.listdir(data_path)
    csv_files = [os.path.join(data_path, c) for c in csv_files if c.endswith(".csv")]

    save_folder = 'testing data'
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
        data = process_data(traj)
        new_traj = data.seperate_data()

        for i in range(5):
            driver = f'driver_{i+1}'
            save_file_name = f'{driver}_{file_name}.pkl'
            save_file_path = os.path.join(save_path, save_file_name)
            data_save = new_traj[driver]
            if len(data_save) == 0:
                print('{} on {} is empty'.format(driver, file_name))
                continue
            with open(save_file_path, 'wb') as fp:
                pickle.dump(new_traj[driver], fp)


#### Validation 
    validation_data = TrajLoader(save_folder)
    batch = 1
    params_v = {'batch_size':batch,'shuffle':False,'num_workers':0}
    data_val = torch.utils.data.DataLoader(validation_data, **params_v)
        
    model_dir = outside_path + "/best_model.pt"
    model = NN_Module.MyLSTM().to(device)
    best_model = torch.load(model_dir)
    model.load_state_dict(best_model["model_state_dict"])

    val_loss, val_acc = NN_Module.testing(model, data_val)
    print('Epoch validation loss: {:.4f}'.format(val_loss))
    print('Epoch validation acc: {:.4f} %'.format(val_acc * 100))


