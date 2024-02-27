import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd
import pickle

method = 'imputation'
default_config = "base_forecasting.yaml" if method=='forecasting' else "base.yaml"
#TODO: use v3 for clean data, v1 for noisy data
datafolder='/home/tompoek/waymo-processed/v1/Selected-Car-Following-CF-pairs-and-their-trajectories'
# datafile='all_segment_paired_car_following_trajectory(position-based, speed-based, processed).csv'
datafile='tail_5segments.csv'
meanstdfile = 'mean_std.pk'
noisy_features = ['position_based_position','position_based_speed','position_based_accer']
clean_features = ['processed_position','processed_speed','processed_accer']

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

if method=='forecasting':
    model = CSDI_Traj_Forecasting(config, args.device).to(args.device)
else:
    model = CSDI_Traj_Imputation(config, args.device).to(args.device)

local_time_np = pd.read_csv(datafolder+'/'+datafile,
                            usecols=['local_time'],
                            ).to_numpy().reshape(-1)
start_segment_idx = 0
segment_id = 1

n_test_segments = 3 # number of segments reserved for testing

with open(datafolder+'/'+meanstdfile, 'rb') as f:
    mean_data, std_data = pickle.load(f)
    mean_data = torch.from_numpy(mean_data).to(args.device).float()
    std_data = torch.from_numpy(std_data).to(args.device).float()

if method=='forecasting':
    evaluator = Evaluator(foldername, args.nsample,
                          mean_scaler=mean_data, scaler=std_data)
else:
    evaluator = Evaluator(foldername, args.nsample)

for local_time_idx in range(1, local_time_np.shape[0]):
    
    if local_time_idx == local_time_np.shape[0]-1:
        local_time_idx += 1
        print(f"Testing at Segment No. {segment_id}")
        test_loader = get_dataloader(
            config["train"]["batch_size"], method=method, device=args.device,
            mode="test",
            start_segment_idx=start_segment_idx, local_time_idx=local_time_idx,
            datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
            noisy_features=noisy_features, clean_features=clean_features
        )
        evaluator.evaluate_segment(model, test_loader, segment_id)

    elif local_time_np[local_time_idx] > local_time_np[local_time_idx-1]:
        continue
    
    else:
        if (local_time_np.shape[0]-local_time_idx) <= 199*(n_test_segments-1):
            print(f"Testing at Segment No. {segment_id}")
            test_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="test",
                start_segment_idx=start_segment_idx, local_time_idx=local_time_idx,
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=noisy_features, clean_features=clean_features
            )
            evaluator.evaluate_segment(model, test_loader, segment_id)
        else:
            print(f"Training at Segment No. {segment_id}")
            train_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="train",
                start_segment_idx=start_segment_idx, local_time_idx=local_time_idx,
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=noisy_features, clean_features=clean_features
            )
            valid_loader = get_dataloader(
                config["train"]["batch_size"], method=method, device=args.device,
                mode="valid",
                start_segment_idx=start_segment_idx, local_time_idx=local_time_idx,
                datafolder=datafolder, datafile=datafile, meanstdfile=meanstdfile,
                noisy_features=noisy_features, clean_features=clean_features
            )
            train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)

        segment_id += 1
        start_segment_idx = local_time_idx

evaluator.save_evaluated_metrics_of_all_segments()

# if args.modelfolder == "":
#     train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)
# else:
#     model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

