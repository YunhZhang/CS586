import os
from tqdm import trange
import pickle
import numpy as np
from torch.utils.data import Dataset, Subset
import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import NN_Module_v1 as NN_Module
from torch.utils.tensorboard import SummaryWriter

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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.file_list[idx]
        with open(file_name, 'rb') as fp:
            data = pickle.load(fp)

        label_driver = data['plate']
        label = np.zeros((5, ))
        label[label_driver] = 1

        assert len(data)>0, 'empty data!'
        lgn = data_expand(data['longitude'], size=self.extend_size)
        lat = data_expand(data['latitude'], size=self.extend_size)
        status = data_expand(np.array(data['status']), size=self.extend_size)
        sec = data_expand(data['seconds'], size=self.extend_size)
        minute = data_expand(data['minutes'], size=self.extend_size)
        hr = data_expand(data['hour'], size=self.extend_size)
        dofw = data_expand(data['day'], size=self.extend_size)
        grid_id = data_expand(data['grid_id'], size=self.extend_size)
        freq_grid = data_expand(data['most_freq_grid'] * np.ones(data['longitude'].shape), size=self.extend_size)
        st_mean = data_expand(data['status_mean'] * np.ones(data['longitude'].shape), size=self.extend_size)
        data_ts = np.hstack((lgn, lat, status, sec, minute, hr, dofw, grid_id, freq_grid, st_mean)).astype('float32')

        return {'data_ts': data_ts, 'labels': label}
    

if __name__ == '__main__':
    folder = 'training data'
    outside_path = os.path.dirname(os.path.realpath(__file__))
    result_folder = os.path.join(outside_path, 'results')
    os.makedirs(result_folder, exist_ok=True)
    dl_writer = SummaryWriter(result_folder)

    data_test = TrajLoader(folder)
    training_data, testing_data = data_split(data_test, 0.2)
    batch_t = 10
    batch_v = 8
    params_t = {'batch_size':batch_t,'shuffle':True,'num_workers':0}
    params_v = {'batch_size':batch_v,'shuffle':True,'num_workers':0}
    data_train = torch.utils.data.DataLoader(training_data, **params_t)
    data_test = torch.utils.data.DataLoader(testing_data, **params_v)

    model = NN_Module.TrajClassification(None).to(device)
    learning_rate = 0.0001 #0.0001
    beta = (0.995, 0.998)
    ep = 1e-10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=beta, eps=ep)

    train_acc_max = 0
    test_acc_max = 0
    train_acc_list = []
    test_acc_list = []

    train_acc_max = 0
    test_acc_max = 0
    train_acc_list = []
    test_acc_list = []
    num_iter_epoch = len(data_train)
    Epoch = 500
    for epoch in trange(Epoch):
        epoch_loss = 0
        epoch_acc = 0
        for i, traj_train in enumerate(data_train):
            data_ts = traj_train['data_ts'].to(device)
            label_gt = traj_train['labels'].to(device)
            if epoch == 0 and i == 0:
                torch.onnx.export(model, data_ts, f"model_origin.onnx", input_names=['input'],
                                  output_names=['output'])
            loss_train = model.training_step(data_ts, label_gt)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            result = model.validation_step(data_ts, label_gt)

            epoch_loss += result['loss']
            epoch_acc += result['acc']

            dl_writer.add_scalar('LossEveryIter', result['loss'], epoch*num_iter_epoch + i)
            dl_writer.add_scalar('TrainAccuracyEveryIter', result['acc'], epoch * num_iter_epoch + i)
            dl_writer.flush()

        epoch_loss = epoch_loss / num_iter_epoch
        epoch_acc = epoch_acc / num_iter_epoch

        val_loss, val_acc = NN_Module.testing(model, data_test)

        print('Epoch train loss: {:.4f}'.format(epoch_loss))
        print('Epoch train acc: {:.4f} %'.format(epoch_acc * 100))
        print('Epoch test loss: {:.4f}'.format(val_loss))
        print('Epoch test acc :{:.4f} %'.format(val_acc * 100))
        save_name = os.path.join(result_folder, f'model_{str(epoch+1).zfill(4)}.ckpt')
        torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss_train}, save_name)

        dl_writer.add_scalar('train_loss', epoch_loss, epoch)
        dl_writer.add_scalar('train_accuracy', epoch_acc, epoch)
        dl_writer.add_scalar('valid_loss', val_loss, epoch)
        dl_writer.add_scalar('valid_acc', val_acc, epoch)
        dl_writer.flush()

        train_acc_max = max(train_acc_max, epoch_acc)
        test_acc_max = max(test_acc_max, val_acc)
        train_acc_list.append(epoch_acc)
        test_acc_list.append(val_acc)

    print('max train acc :{:.4f} %'.format(train_acc_max * 100))
    print('max test acc :{:.4f} %'.format(test_acc_max * 100))
#     Epoch = 1
#     k = 0
#     df = pd.DataFrame(columns=['Epoch','iteration','train loss','train acc','test loss', 'test acc'])

#     for epoch in trange(Epoch):
#         for i, performance_ in enumerate(data_train):
#             features = performance_['data_ts'].to(device)
#             label = performance_['labels'].to(device)
#             # if epoch == 0 and i == 0:
#             #     torch.onnx.export(model, features, f"model_origin.onnx", input_names=['input'],
#             #                       output_names=['output'])
#             train_loss, train_acc = model.training_step(features, label)

#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             if k % 100 == 0:
#                 loss_val = []
#                 acc_val = []
#                 for i_val, performance_v_ in enumerate(data_test):
#                     features_v = performance_v_['data_ts'].to(device)
#                     label_v = performance_v_['labels'].to(device)
#                     test_loss, test_acc = model.validation_step(features, label)
#                     loss_val.append(test_loss)
#                     acc_val.append(test_acc)
#                 loss_val = sum(loss_val)/len(loss_val)
#                 acc_val = sum(acc_val)/len(acc_val)
                
#                 output_path = result_folder+'/'+str(k)+'/'
#                 os.makedirs(output_path,exist_ok=True)
#                 torch.save(model, output_path +'unet.pt')
#                 if k > 100:
#                     df.to_csv(output_path + 'training_record.csv')  
#             k += 1
#             print(['Epoch','iteration','train loss','train acc','test loss', 'test acc'])
#             print(epoch,k,train_loss.item(),train_acc.item(), test_loss, test_acc.item())
#             df[k] = ['Epoch','iteration','train loss','train acc','test loss', 'test acc']
# df.to_csv(result_folder + '/training_record.csv')        
