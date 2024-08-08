import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

method = 'forecasting'
default_config = "base_forecasting.yaml" if method=='forecasting' else "base.yaml"
datafolder='/home/ubuntu/waymo-processed'
datafile='all_segments.csv'
meanstdfile = 'mean_std.pk'
v1_noisy_features = ['position_based_position','position_based_speed','position_based_accer']
v3_clean_features = ['filter_pos','filter_speed','filter_accer']
id_columns = ['segment_id','local_veh_id','follower_id','leader_id']
time_column = ['local_time']
locator_columns = id_columns+time_column

from dataset_trajectory import get_dataloader
from main_model import CSDI_Traj_Imputation, CSDI_Traj_Forecasting
from utils import train, evaluate, Evaluator

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default=default_config)
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument("--nsample", type=int, default=5)
parser.add_argument("--unconditional", action="store_true")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
if method=='forecasting':
    foldername = ("./save/traj_forecasting" + "_" + current_time + "/")
else:
    foldername = ("./save/traj_imputation" + "_" + current_time + "/")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

with open(datafolder+'/'+meanstdfile, 'rb') as f:
    mean_data, std_data = pickle.load(f)
    mean_data = torch.from_numpy(mean_data).to(args.device).float()
    std_data = torch.from_numpy(std_data).to(args.device).float()

if method=='forecasting':
    evaluator = Evaluator(foldername, args.nsample,
                          mean_scaler=mean_data, scaler=std_data)
else:
    evaluator = Evaluator(foldername, args.nsample)


random_state = 100
full_dataset_ids = pd.read_csv(datafolder+'/'+datafile, usecols=id_columns).drop_duplicates()
train_ids = pd.read_csv(datafolder+'/'+'train_ids_random_state_'+str(random_state)+'_split9to1.csv')
test_ids = pd.read_csv(datafolder+'/'+'test_ids_random_state_'+str(random_state)+'_split9to1.csv')

modelfolder = '' # or 'traj_forecasting_20240731_172548'
do_training = True
do_testing = True

if modelfolder == '':
    model = CSDI_Traj_Forecasting(config, args.device).to(args.device) if method=='forecasting' else CSDI_Traj_Imputation(config, args.device).to(args.device)
else:
    model = torch.load('./save/'+modelfolder+'/model.pth').to(args.device)

if do_training:
    for ids in full_dataset_ids.iterrows(): # train model
            print(f"Training at Segment No. {ids[1]['segment_id']} Local Vehicle ID: {ids[1]['local_veh_id']}")
            train_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="train",
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=v1_noisy_features, clean_features=v3_clean_features,
                id_columns=id_columns, ids=ids[1]
            )
            valid_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="valid",
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=v1_noisy_features, clean_features=v3_clean_features,
                id_columns=id_columns, ids=ids[1]
            )
            train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)

if do_testing:
    for ids in test_ids.iterrows(): # test model
            print(f"Testing at Segment No. {ids[1]['segment_id']} Local Vehicle ID: {ids[1]['local_veh_id']}")
            test_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="test",
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=v1_noisy_features, clean_features=v3_clean_features,
                id_columns=id_columns, ids=ids[1]
            )
            evaluator.evaluate_segment(model, test_loader, ids[1]['segment_id'], ids[1]['local_veh_id'])

evaluator.save_evaluated_metrics_of_all_segments()
