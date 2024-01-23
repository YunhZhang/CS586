import os
from glob import glob

dynamic_path = os.path.abspath(__file__+"/../")
print(dynamic_path)

def file_list_load(load_path):
    file_list = sorted(glob(os.path.join(load_path, '*.csv'.format(iter))),
                       key=lambda x: (int(x.split('/')[-1].split('.')[0].split('_')[-2]),
                                      int(x.split('/')[-1].split('.')[0].split('_')[-1])))
    return file_list

data_path = os.path.join(dynamic_path, 'data_5drivers')
file_names = file_list_load(data_path)


a =1