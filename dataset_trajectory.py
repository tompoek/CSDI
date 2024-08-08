import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Traj_Forecasting_Dataset(Dataset):
    def __init__(self,
                 mode,
                 datafolder,
                 datafile,
                 meanstdfile,
                 noisy_features,
                 clean_features,
                 id_columns,
                 ids,
                 validindex=0,
                 mask_percentage=0.5):

        self.pred_length = 10
        self.history_length = self.pred_length*4

        self.valid_length = self.pred_length*2

        df = pd.read_csv(datafolder+'/'+datafile,
                        usecols=id_columns+['local_time']+noisy_features,
                        )
        for id_col in id_columns:
            df = df[df[id_col]==ids[id_col]]
        df.drop(labels=id_columns,axis=1,inplace=True)
        df[['position_based_position']] -= df[['position_based_position']].iloc[0] # set all segments' initial position to 0
        self.main_data = df[noisy_features].to_numpy()
        mask = np.zeros(self.main_data.shape[0])
        mask[:int(mask_percentage*self.main_data.shape[0])] = 1
        np.random.Generator(np.random.PCG64(seed=42)).shuffle(mask) # this will make the mask reproducible
        self.mask_data = np.repeat(mask.reshape(-1,1),repeats=self.main_data.shape[1],axis=1)
        with open(datafolder+'/'+meanstdfile, 'rb') as f:
            self.mean_data, self.std_data = pickle.load(f)
            
        self.seq_length = self.history_length + self.pred_length
        self.main_data = (self.main_data - self.mean_data) / self.std_data

        if mode == 'train':
            self.use_index = np.arange(0, len(self.main_data)-self.seq_length, 1)
        elif mode == 'valid':
            self.use_index = np.arange(validindex, validindex+self.valid_length, self.pred_length)
        else: #test
            self.use_index = np.arange(0, len(self.main_data)-self.seq_length, self.pred_length)
        
    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }

        return s
    def __len__(self):
        return len(self.use_index)


class Traj_Imputation_Dataset(Dataset):
    def __init__(self,
                 mode,
                 datafolder,
                 datafile,
                 noisy_features,
                 clean_features,
                 id_columns,
                 ids,
                 eval_length=20, target_dim=3):
        self.a_min = -8
        self.a_max = 5
        self.time_window = 20 # in 0.1s
        #TODO: change eval_length to time_window (currently only supports eval_length=10)
        self.eval_length = eval_length
        self.target_dim = target_dim

        #TODO: safely remove legacy mask codes from pm2.5 implementation
        # if mode == "train":
        #     second_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        #     # seconds excluded from histmask (since they are used for creating missing patterns in test dataset)
        #     flag_for_histmask = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        #     second_list.pop(validindex)
        #     flag_for_histmask.pop(validindex)
        # elif mode == "valid":
        #     second_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        #     second_list = second_list[validindex : validindex + 1]
        # elif mode == "test":
        #     second_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        # self.second_list = second_list

        # create data for batch
        # self.observed_data = []  # values (separated into each month)
        # self.observed_mask = []  # masks (separated into each month)
        # self.gt_mask = []  # ground-truth masks (separated into each month)
        # self.index_month = []  # indicate month
        # self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        # self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        # self.cut_length = []  # excluded from evaluation targets

        df = pd.read_csv(datafolder+'/'+datafile,
                         usecols=id_columns+['local_time']+clean_features,
                         )
        df_gt = pd.read_csv(datafolder+'/'+datafile,
                            usecols=id_columns+noisy_features,
                            )
        for id_col in id_columns:
            df = df[df[id_col]==ids[id_col]]
            df_gt = df_gt[df_gt[id_col]==ids[id_col]]
        df.drop(labels=id_columns,axis=1,inplace=True)
        df_gt.drop(labels=id_columns,axis=1,inplace=True)

        #TODO: generate noises randomly, instead of using outliers detection
        accel_too_low = (df_gt['position_based_accer'] < self.a_min).values
        accel_too_high = (df_gt['position_based_accer'] > self.a_max).values
        noisy_points = np.any([accel_too_low,accel_too_high],axis=0)
        noisy_time_windows = noisy_points
        # detect outliers within +/- half of time window
        for i in np.where(noisy_points)[0]:
            noisy_time_windows[max(i-int(self.time_window/2),0) : min(i+int(self.time_window/2),noisy_time_windows.shape[0])] = True
        # df.loc[noisy_time_windows, clean_features] = np.nan
        df_gt.loc[noisy_time_windows, noisy_features] = np.nan

        start_date_hour_minute = '2017-01-01T00:00:' # arbitrarily set a start time
        df['datetime'] = np.datetime64(start_date_hour_minute+'00.1') + pd.to_timedelta(df['local_time'], unit='s')
        df.drop(['local_time'],axis=1,inplace=True)

        df_gt['datetime'] = df['datetime']
        df.set_index('datetime',inplace=True)
        df_gt.set_index('datetime',inplace=True)
        
        # for i in range(len(second_list)):
        #     current_df = df[df.index.second == second_list[i]]
        #     current_df_gt = df_gt[df_gt.index.second == second_list[i]]
        #     current_length = len(current_df) - eval_length + 1

        #     last_index = len(self.index_month)
        #     self.index_month += np.array([i] * current_length).tolist()
        #     self.position_in_month += np.arange(current_length).tolist()
        #     if mode == "train":
        #         self.valid_for_histmask += np.array(
        #             [flag_for_histmask[i]] * current_length
        #         ).tolist()

        #     # mask values for observed indices are 1
        #     c_mask = 1 - current_df.isnull().values
        #     c_gt_mask = 1 - current_df_gt.isnull().values
        #     c_data = (
        #         current_df.fillna(0).values
        #     ) * c_mask

        #     self.observed_mask.append(c_mask)
        #     self.gt_mask.append(c_gt_mask)
        #     self.observed_data.append(c_data)

        #     if mode == "test":
        #         n_sample = len(current_df) // eval_length
        #         # interval size is eval_length (missing values are imputed only once)
        #         c_index = np.arange(
        #             last_index, last_index + eval_length * n_sample, eval_length
        #         )
        #         self.use_index += c_index.tolist()
        #         self.cut_length += [0] * len(c_index)
        #         if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
        #             self.use_index += [len(self.index_month) - 1]
        #             self.cut_length += [eval_length - len(current_df) % eval_length]
                    
        length_excl_eval = len(df) - eval_length + 1

        # last_index = len(self.index_month)
        # self.index_month += np.array([0] * length_excl_eval).tolist()
        # self.position_in_month += np.arange(length_excl_eval).tolist()

        # mask values for observed indices are 1
        self.observed_mask = 1 - df.isnull().values
        self.gt_mask = 1 - df_gt.isnull().values
        self.observed_data = (df.fillna(0).values) * self.observed_mask

        if mode == "test":
            n_sample = len(df) // eval_length
            # interval size is eval_length (missing values are imputed only once)
            c_index = np.arange(
                0, 0 + eval_length * n_sample, eval_length
            )
            self.use_index += c_index.tolist()
            if len(df) % eval_length != 0:  # avoid double-count for the last time-series
                self.use_index += [length_excl_eval - 1]

        if mode != "test":
            self.use_index = np.arange(length_excl_eval)
            # self.cut_length = [0] * len(self.use_index)

        # if mode == "train":
        #     ind = -1
        #     self.index_month_histmask = []
        #     self.position_in_month_histmask = []

        #     for i in range(len(self.index_month)):
        #         while True:
        #             ind += 1
        #             if ind == len(self.index_month):
        #                 ind = 0
        #             if self.valid_for_histmask[ind] == 1:
        #                 self.index_month_histmask.append(self.index_month[ind])
        #                 self.position_in_month_histmask.append(
        #                     self.position_in_month[ind]
        #                 )
        #                 break
        # else:  # dummy (histmask is only used for training)
        #     self.index_month_histmask = self.index_month
        #     self.position_in_month_histmask = self.position_in_month

        df.drop(labels=df.columns,axis=1,inplace=True)
        df.drop(index=df.index,inplace=True)
        df_gt.drop(labels=df_gt.columns,axis=1,inplace=True)
        df_gt.drop(index=df_gt.index,inplace=True)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        # c_month = self.index_month[index]
        # c_index = self.position_in_month[index]
        # hist_month = self.index_month_histmask[index]
        # hist_index = self.position_in_month_histmask[index]
        s = {
            "observed_data": self.observed_data[
                index : index + self.eval_length
            ],
            "observed_mask": self.observed_mask[
                index : index + self.eval_length
            ],
            "gt_mask": self.gt_mask[
                index : index + self.eval_length
            ],
            # "hist_mask": self.observed_mask[hist_month][
            #     hist_index : hist_index + self.eval_length
            # ],
            "timepoints": np.arange(self.eval_length),
            # "cut_length": self.cut_length[org_index],
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size, method, device, mode,
                   datafolder, datafile, meanstdfile, noisy_features, clean_features,
                   id_columns, ids,
                   validindex=0):
    
    shuffle = True if mode=="train" else False
    if method=='forecasting':
        dataset = Traj_Forecasting_Dataset(mode=mode,
                                           datafolder=datafolder, datafile=datafile,
                                           meanstdfile=meanstdfile,
                                           noisy_features=noisy_features, clean_features=clean_features,
                                           id_columns=id_columns, ids=ids)
    else:
        dataset = Traj_Imputation_Dataset(mode=mode,
                                          datafolder=datafolder, datafile=datafile,
                                          noisy_features=noisy_features, clean_features=clean_features,
                                          id_columns=id_columns, ids=ids)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle
    )

    return data_loader
