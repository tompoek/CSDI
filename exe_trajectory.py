import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd

method = 'imputation'
default_config = "base_forecasting.yaml" if method=='forecasting' else "base.yaml"
datafolder='/home/tompoek/waymo-processed/v1/Selected-Car-Following-CF-pairs-and-their-trajectories'
datafile='tail_5segments.csv'
noisy_features = ['position_based_position','position_based_speed','position_based_accer']
clean_features = ['processed_position','processed_speed','processed_accer']

from dataset_trajectory import get_dataloader
from main_model import CSDI_Traj_Imputation, CSDI_Traj_Forecasting
from utils import train, evaluate

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
segment_number = 1

for local_time_idx in range(1, local_time_np.shape[0]):
    
    if local_time_idx == local_time_np.shape[0]-1:
        local_time_idx += 1
        test_loader, mean_data, std_data = get_dataloader(
            config["train"]["batch_size"], method=method, device=args.device,
            mode="test",
            start_segment_idx=start_segment_idx, local_time_idx=local_time_idx
        )

    elif local_time_np[local_time_idx] > local_time_np[local_time_idx-1]:
        continue
    
    else:
        print(f"Segment No. {segment_number}")
        train_loader, mean_data, std_data = get_dataloader(
            config["train"]["batch_size"], method=method, device=args.device,
            mode="train",
            start_segment_idx=start_segment_idx, local_time_idx=local_time_idx
        )
        valid_loader, mean_data, std_data = get_dataloader(
            config["train"]["batch_size"], method=method, device=args.device,
            mode="valid",
            start_segment_idx=start_segment_idx, local_time_idx=local_time_idx
        )

        train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)

        segment_number += 1
        start_segment_idx = local_time_idx

# if args.modelfolder == "":
#     train(model, config["train"], train_loader, valid_loader=valid_loader, foldername=foldername)
# else:
#     model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

if method=='forecasting':
    evaluate(model, test_loader, nsample=args.nsample, foldername=foldername,
             mean_scaler=mean_data, scaler=std_data)
else:
    evaluate(model, test_loader, nsample=args.nsample, foldername=foldername)
