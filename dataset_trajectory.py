import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Traj_Forecasting_Dataset(Dataset):
    def __init__(self,
                 start_segment_idx, local_time_idx,
                 datafolder='/home/tompoek/waymo-processed/v1/Selected-Car-Following-CF-pairs-and-their-trajectories',
                 datafile='all_segment_paired_car_following_trajectory(position-based, speed-based, processed).csv',
                 noisy_features=['position_based_position','position_based_speed','position_based_accer'],
                 clean_features=['processed_position','processed_speed','processed_accer'],
                 validindex=0,
                 mask_percentage=0.5,
                 mode="train"):

        datatype = 'trajectory'
        if datatype == 'electricity':
            self.history_length = 24*7
            self.pred_length = 24

            datafolder = './data/electricity_nips'
            self.test_length = 24*7
            self.valid_length = 24*5

            paths=datafolder+'/data.pkl' 
            #shape: (T x N)
            #mask_data is usually filled by 1
            with open(paths, 'rb') as f:
                self.main_data, self.mask_data = pickle.load(f)
            paths=datafolder+'/meanstd.pkl'
            with open(paths, 'rb') as f:
                self.mean_data, self.std_data = pickle.load(f)

            subset_length = 24*22
            feature_length = 3
            self.main_data = self.main_data[:subset_length,:feature_length]
            self.mask_data = self.mask_data[:subset_length,:feature_length]
            self.mean_data = self.mean_data[:feature_length]
            self.std_data = self.std_data[:feature_length]
        else:
            self.pred_length = 10
            self.history_length = self.pred_length*4

            self.valid_length = self.pred_length*2

            df = pd.read_csv(datafolder+'/'+datafile,
                            usecols=['local_time']+clean_features,
                            skiprows=range(1,1+start_segment_idx), nrows=local_time_idx-start_segment_idx,
                            )
            df[['processed_position']] -= df[['processed_position']].iloc[0] # set all segments' initial position to 0
            self.main_data = df[clean_features].to_numpy()
            mask = np.zeros(self.main_data.shape[0])
            mask[:int(mask_percentage*self.main_data.shape[0])] = 1
            np.random.Generator(np.random.PCG64(seed=42)).shuffle(mask)
            self.mask_data = np.repeat(mask.reshape(-1,1),repeats=self.main_data.shape[1],axis=1)
            meanstdfile = 'mean_std.pk'
            with open(datafolder+'/'+meanstdfile, 'rb') as f:
                self.mean_data, self.std_data = pickle.load(f)
            
        self.seq_length = self.history_length + self.pred_length
        self.main_data = (self.main_data - self.mean_data) / self.std_data


        if datatype=='electricity':
            total_length = len(self.main_data)
            if mode == 'train': 
                start = 0
                end = total_length - self.seq_length - self.valid_length - self.test_length + 1
                self.use_index = np.arange(start,end,1)
            if mode == 'valid': #valid
                start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
                end = total_length - self.seq_length - self.test_length + self.pred_length
                self.use_index = np.arange(start,end,self.pred_length)
            if mode == 'test': #test
                start = total_length - self.seq_length - self.test_length + self.pred_length
                end = total_length - self.seq_length + self.pred_length
                self.use_index = np.arange(start,end,self.pred_length)
        else:
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
                 start_segment_idx, local_time_idx,
                 datafolder='~/waymo-processed/v1/Selected-Car-Following-CF-pairs-and-their-trajectories',
                 datafile='all_segment_paired_car_following_trajectory(position-based, speed-based, processed).csv',
                 noisy_features=['position_based_position','position_based_speed','position_based_accer'],
                 clean_features=['processed_position','processed_speed','processed_accer'],
                 eval_length=10, target_dim=3, validindex=0,
                 mode="train"):
        self.eval_length = eval_length
        self.target_dim = target_dim

        if mode == "train":
            second_list = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            # seconds excluded from histmask (since they are used for creating missing patterns in test dataset)
            flag_for_histmask = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
            second_list.pop(validindex)
            flag_for_histmask.pop(validindex)
        elif mode == "valid":
            second_list = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
            second_list = second_list[validindex : validindex + 1]
        elif mode == "test":
            second_list = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        self.second_list = second_list

        # create data for batch
        self.observed_data = []  # values (separated into each month)
        self.observed_mask = []  # masks (separated into each month)
        self.gt_mask = []  # ground-truth masks (separated into each month)
        self.index_month = []  # indicate month
        self.position_in_month = []  # indicate the start position in month (length is the same as index_month)
        self.valid_for_histmask = []  # whether the sample is used for histmask
        self.use_index = []  # to separate train/valid/test
        self.cut_length = []  # excluded from evaluation targets

        # df = pd.read_csv('~/waymo-processed/v3/all_seg_paired_cf_trj_final.csv',
        #                  usecols=['local_time','filter_pos','filter_speed','filter_accer'],
        #                  skiprows=range(1,1+start_segment_idx), nrows=local_time_idx-start_segment_idx,
        #                  )
        df = pd.read_csv(datafolder+'/'+datafile,
                         usecols=['local_time']+clean_features,
                         skiprows=range(1,1+start_segment_idx), nrows=local_time_idx-start_segment_idx,
                         )
        start_date_hour_minute = '2017-01-01T00:00:' # arbitrarily set a start time
        df['datetime'] = np.datetime64(start_date_hour_minute+'00.1') + pd.to_timedelta(df['local_time'], unit='S')
        # df['datetime'] = np.datetime64(start_date_hour_minute+'00.1')
        # df['local_time'] = pd.to_timedelta(df['local_time'], unit='S')
        # start_segment_idx = 0
        # for i in range(1, df.shape[0]):
        #     if df.loc[i,'local_time'] > df.loc[i-1,'local_time']:
        #         df.loc[i,'datetime'] = df.loc[start_segment_idx,'datetime'] + df.loc[i,'local_time']
        #     else:
        #         df.loc[i,'datetime'] = df.loc[start_segment_idx,'datetime'] + np.timedelta64(1,'D')
        #         start_segment_idx = i
        
        df.drop(['local_time'],axis=1,inplace=True)
        df.set_index('datetime',inplace=True)

        df_gt = df.copy()
        if mode == "test":
            for i in range(0,10):
                # df_gt.loc[start_date_hour_minute+'0'+str(i)+'.3':start_date_hour_minute+'0'+str(i)+'.5',['processed_accer']] = np.nan
                # df_gt.loc[start_date_hour_minute+'0'+str(i)+'.4':start_date_hour_minute+'0'+str(i)+'.6',['processed_speed']] = np.nan
                # df_gt.loc[start_date_hour_minute+'0'+str(i)+'.5':start_date_hour_minute+'0'+str(i)+'.7',['processed_position']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+'0'+str(i)+'.3')*(df_gt.index<start_date_hour_minute+'0'+str(i)+'.5'),
                          ['processed_accer']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+'0'+str(i)+'.4')*(df_gt.index<start_date_hour_minute+'0'+str(i)+'.6'),
                          ['processed_speed']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+'0'+str(i)+'.5')*(df_gt.index<start_date_hour_minute+'0'+str(i)+'.7'),
                          ['processed_position']] = np.nan
            for i in range(10,20):
                # df_gt.loc[start_date_hour_minute+str(i)+'.3':start_date_hour_minute+str(i)+'.5',['processed_accer']] = np.nan
                # df_gt.loc[start_date_hour_minute+str(i)+'.4':start_date_hour_minute+str(i)+'.6',['processed_speed']] = np.nan
                # df_gt.loc[start_date_hour_minute+str(i)+'.5':start_date_hour_minute+str(i)+'.7',['processed_position']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+str(i)+'.3')*(df_gt.index<start_date_hour_minute+str(i)+'.5'),
                          ['processed_accer']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+str(i)+'.4')*(df_gt.index<start_date_hour_minute+str(i)+'.6'),
                          ['processed_speed']] = np.nan
                df_gt.loc[(df_gt.index>start_date_hour_minute+str(i)+'.5')*(df_gt.index<start_date_hour_minute+str(i)+'.7'),
                          ['processed_position']] = np.nan

        for i in range(len(second_list)):
            current_df = df[df.index.second == second_list[i]]
            current_df_gt = df_gt[df_gt.index.second == second_list[i]]
            current_length = len(current_df) - eval_length + 1

            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            # mask values for observed indices are 1
            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_data = (
                current_df.fillna(0).values
            ) * c_mask

            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            if mode == "test":
                n_sample = len(current_df) // eval_length
                # interval size is eval_length (missing values are imputed only once)
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:  # avoid double-count for the last time-series
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        # masks for 1st,4th,7th,10th months are used for creating missing patterns in test data,
        # so these months are excluded from histmask to avoid leakage
        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []

            for i in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:  # dummy (histmask is only used for training)
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month

        df.drop(labels=df.columns,axis=1,inplace=True)
        df.drop(index=df.index,inplace=True)
        df_gt.drop(labels=df_gt.columns,axis=1,inplace=True)
        df_gt.drop(index=df_gt.index,inplace=True)

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]
        hist_month = self.index_month_histmask[index]
        hist_index = self.position_in_month_histmask[index]
        s = {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "gt_mask": self.gt_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ],
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size, method, device, mode, start_segment_idx, local_time_idx, validindex=0):
    
    shuffle = True if mode=="train" else False
    if method=='forecasting':
        dataset = Traj_Forecasting_Dataset(mode=mode,
                                start_segment_idx=start_segment_idx, local_time_idx=local_time_idx)
        mean_data = torch.from_numpy(dataset.mean_data).to(device).float()
        std_data = torch.from_numpy(dataset.std_data).to(device).float()
    else:
        dataset = Traj_Imputation_Dataset(mode=mode, validindex=validindex,
                                start_segment_idx=start_segment_idx, local_time_idx=local_time_idx)
        mean_data = torch.zeros(dataset.target_dim).to(device).float()
        std_data = torch.ones(dataset.target_dim).to(device).float()
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle
    )

    return data_loader, mean_data, std_data
