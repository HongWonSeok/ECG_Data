import os
import json
import torch
import argparse

# from trainer.trainer import ECGClassifierTrainer
from trainer.trainer import ECGClassifierTrainer



def main():

    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) 
    args.update(param)
    
    ECGClassifierTrainer(args).train()
    

def load_json(path):
    with open(path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser   = argparse.ArgumentParser(description='ECG data classification')
    
    parser.add_argument('--config', default='./configs/config_Icentia_batch.json')
   
    return parser


if __name__ =="__main__":
    
    main()
