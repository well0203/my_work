import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        # If you want further debug it it will be helpful :)
        self.border1s = None  # Initialize border1s attribute
        self.border2s = None  # Initialize border2s attribute
        self.border1 = None  # Initialize border1 attribute
        self.border2 = None  # Initialize border2 attribute

    def __read_data__(self):
        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.features == 'S':
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
        else:
            cols = list(df_raw.columns)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]


        # Same as for ARIMA preprocessing
        # number of days 
        train_size = int(round(len(df_raw)/24*0.7, 0))
        test_size = int(round(len(df_raw)/24*0.15, 0))

        # calculate number of observations in each dataset
        num_train = train_size*24
        num_test = test_size*24
        num_vali = len(df_raw) - num_train - num_test 


        self.border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        self.border2s = [num_train, num_train + num_vali, len(df_raw)]
        self.border1 = self.border1s[self.set_type]
        self.border2 = self.border2s[self.set_type]

        if self.set_type == 0:
            # I want to choose the latest samples, they chose the first
            # Moreover, they do not have a percentage of dataset, rather some type
            # of adjustment to the number of series
            tmp = self.border2*self.percent // 100
            self.border1 = self.border2 - tmp

            if tmp <= self.seq_len:
                raise ValueError(f'The percent of training data {tmp} should be larger than the sequence length {self.seq_len}. Please choose a larger percent.')
            # border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[self.border1s[0]:self.border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][self.border1:self.border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[self.border1:self.border2]
        self.data_y = data[self.border1:self.border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
