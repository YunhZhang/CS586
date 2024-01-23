import os
from tqdm import trange
import pickle
import numpy as np
from torch.utils.data import Dataset, Subset
import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import NN_Module_v2 as NN_Module


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mkdir(folder_name):
    outside_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(outside_path, folder_name)
    pkl_files = os.listdir(data_path)
    pkl_files = [os.path.join(data_path, c) for c in pkl_files]
    return pkl_files

def data_split(data, ratio):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=ratio, shuffle=True)
    train_data = Subset(data, train_idx)
    test_data = Subset(data, val_idx)
    return train_data, test_data

def data_expand(array, size=10000):
    #expand data to (10000,1)
    array = array.reshape(-1,)
    output = np.zeros((size, 1))
    output[:len(array), 0] = array
    return output

class TrajLoader(Dataset):
    def __init__(self, folder_name, extend_size=10000, transform=None):
        self.folder_name = folder_name
        self.transform = transform
        self.file_list = mkdir(folder_name)
        self.extend_size = extend_size

    def __len__(self, num=1):
        return len(self.file_list)*num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_list[idx]
        with open(file_name, 'rb') as fp:
            data = pickle.load(fp)

        if len(data['status_0']) > 0:
            label_driver = data['status_0']['plate']
        elif len(data['status_1']) > 0:
            label_driver = data['status_1']['plate']
        # else:
            # print('Error! Both status are empty')
        label = np.zeros((5, ))
        label[label_driver] = 1

        if len(data['status_0']) > 0:
            lgn = data_expand(data['status_0']['longitude'], size=self.extend_size)
            lat = data_expand(data['status_0']['latitude'], size=self.extend_size)
            status = data_expand(np.array(data['status_0']['status']), size=self.extend_size)
            sec = data_expand(data['status_0']['seconds'], size=self.extend_size)
            minute = data_expand(data['status_0']['minutes'], size=self.extend_size)
            hr = data_expand(data['status_0']['hour'], size=self.extend_size)
            # date = data_expand(data['status_0']['date'], size=self.extend_size)
            # month = data_expand(data['status_0']['month'], size=self.extend_size)
            dofw = data_expand(data['status_0']['day'], size=self.extend_size)
            grid_id = data_expand(data['status_0']['grid_id'], size=self.extend_size)
            freq_grid = data_expand(data['status_0']['most_freq_grid'] * np.ones(data['status_0']['longitude'].shape), size=self.extend_size)
            # st_mean = data_expand(data['status_0']['status_mean'] * np.ones(data['longitude'].shape), size=self.extend_size)
            data_ts_0 = np.hstack((lgn, lat, status, sec, minute, hr, dofw, grid_id, freq_grid)).astype('float32')
        else:
            data_ts_0 = np.zeros((self.extend_size, 11)).astype('float32')

        if len(data['status_1']) > 0:
            lgn = data_expand(data['status_1']['longitude'], size=self.extend_size)
            lat = data_expand(data['status_1']['latitude'], size=self.extend_size)
            status = data_expand(np.array(data['status_1']['status']), size=self.extend_size)
            sec = data_expand(data['status_1']['seconds'], size=self.extend_size)
            minute = data_expand(data['status_1']['minutes'], size=self.extend_size)
            hr = data_expand(data['status_1']['hour'], size=self.extend_size)
            # date = data_expand(data['status_1']['date'], size=self.extend_size)
            # month = data_expand(data['status_1']['month'], size=self.extend_size)
            dofw = data_expand(data['status_1']['day'], size=self.extend_size)
            grid_id = data_expand(data['status_1']['grid_id'], size=self.extend_size)
            freq_grid = data_expand(data['status_1']['most_freq_grid'] * np.ones(data['status_1']['longitude'].shape), size=self.extend_size)
            # st_mean = data_expand(data['status_1']['status_mean'] * np.ones(data['longitude'].shape), size=self.extend_size)
            data_ts_1 = np.hstack((lgn, lat, status, sec, minute, hr, dofw, grid_id, freq_grid)).astype('float32')
        else:
            data_ts_1 = np.zeros((self.extend_size, 11)).astype('float32')

        return {'data_s0': data_ts_0, 'data_s1': data_ts_1, 'labels': label}

if __name__ == '__main__':
    folder = 'training data separate'
    outside_path = os.path.dirname(os.path.realpath(__file__))
    result_folder = os.path.join(outside_path, 'results_v2_9features')
    os.makedirs(result_folder, exist_ok=True)

    data_test = TrajLoader(folder)
    training_data, testing_data = data_split(data_test, 0.2)
    batch_t = 10
    batch_v = 8
    params_t = {'batch_size':batch_t,'shuffle':True,'num_workers':0}
    params_v = {'batch_size':batch_v,'shuffle':False,'num_workers':0}
    data_train = torch.utils.data.DataLoader(training_data, **params_t)
    data_test = torch.utils.data.DataLoader(testing_data, **params_v)

    model = NN_Module.MyLSTM().to(device)
    learning_rate = 0.0001 #0.0001
    beta = (0.995, 0.998)
    ep = 1e-10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=beta, eps=ep)

    train_acc_max = 0
    test_acc_max = 0
    num_iter_train = len(data_train)

    df = pd.DataFrame(columns=['Epoch','training_loss','training_acc','validation_loss','validation_acc'])
    Epoch = 600
    for epoch in trange(Epoch):
        model.train()
        training_loss = 0
        trainig_acc = 0
        for i, traj_train_ in enumerate(data_train):
            traj_train0 = traj_train_['data_s0'].to(device)
            traj_train1 = traj_train_['data_s1'].to(device)
            label_gt = traj_train_['labels'].to(device)

            output = model(traj_train0,traj_train1)
            loss_train = NN_Module.training_loss(output, label_gt)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            result = NN_Module.validation(output, label_gt)

            training_loss += result['loss']
            trainig_acc += result['acc']

        training_loss = training_loss / num_iter_train
        trainig_acc = trainig_acc / num_iter_train

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            iter = len(data_test)
            for i, traj_test in enumerate(data_test):
                traj_test0 = traj_test['data_s0'].to(device)
                traj_test1 = traj_test['data_s1'].to(device)
                label_gt_test = traj_test['labels'].to(device)

                output_val = model(traj_test0, traj_test1)
                result_val = NN_Module.validation(output_val,label_gt_test)

                val_loss += result_val['loss']
                val_acc += result_val['acc']
            val_loss = val_loss / iter
            val_acc = val_acc / iter

        print('Epoch train loss: {:.4f}'.format(training_loss))
        print('Epoch train acc: {:.4f} %'.format(trainig_acc * 100))
        print('Epoch test loss: {:.4f}'.format(val_loss))
        print('Epoch test acc :{:.4f} %'.format(val_acc * 100))
        save_name = os.path.join(result_folder, f'model_{str(epoch+1).zfill(4)}.pt')
        torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss_train}, save_name)

        train_acc_max = max(train_acc_max, trainig_acc)
        test_acc_max = max(test_acc_max, val_acc)

        df.loc[epoch] = [epoch, training_loss, trainig_acc, val_loss, val_acc]
        
    print('max train acc :{:.4f} %'.format(train_acc_max * 100))
    print('max test acc :{:.4f} %'.format(test_acc_max * 100))
    df.to_csv(result_folder+'training_record.csv')