from typing import Tuple, List, Optional

from data.data_loader import IcentiaLoader_MLP, IcentiaLoader_MLP_batch, CinCLoader
import torch.utils.data as data
from torch.utils.data import random_split
from data.augmentations import Compose, RandomNoise, RandomShift, AugmentationPipeline
from configs.config import TRAINING_SPLIT_CHALLANGE, VALIDATION_SPLIT_CHALLANGE, AUGMENTATION_PIPELINE_CONFIG_2C
import dataclasses
import os
import numpy as np
import csv
import scipy.io as sio
from enum import Enum
from typing import Dict, Any
import torch


class Mode(Enum):
    train = "train"
    eval = "eval"

def get_data_loaders(data_args, model_args):
    
  
    path = data_args['args']['path']
    transforms = get_trans()       
            
    if data_args['args']['class'] == 'binary':  
        

        if data_args['type'] == "Icentia":
            print("DataSet : Icentia")
            
        
            
            if model_args['type'] == 'MLP':
                return {
                    Mode.train: data.DataLoader(
                        IcentiaLoader_MLP(path['train_signal'],path['train_label'], transforms[Mode.train]),
                        batch_size=data_args['args']['batch_size'],
                        num_workers=data_args['args']['num_workers'],
                        shuffle=True,
                        drop_last=True
                    ),
                    Mode.eval: data.DataLoader(
                        IcentiaLoader_MLP(path['eval_signal'],path['eval_label'], transforms[Mode.eval]),
                        batch_size=data_args['args']['batch_size'],
                        num_workers=data_args['args']['num_workers'],
                        shuffle=False,
                        drop_last=True
                    )
                }
                
        elif data_args['type'] == "Icentia_batch":
                print("DataSet : Icentia_batch")
                
                TRAINING_SPLIT_ICENTIA11K = list(range(10000))
                VALIDATION_SPLIT_ICENTIA11K = list(range(10000,11000))
                
                if data_args['args']['signal_type'] == 'beat':
                    crop = (250,500)
                    ecg_length = 500
                elif data_args['args']['signal_type'] == 'rythm':
                    crop = (9000,18000)
                    ecg_length = 18000
                
                if model_args['type'] == 'MLP':
                    return {
                        Mode.train: data.DataLoader(
                            IcentiaLoader_MLP_batch(path,split = TRAINING_SPLIT_ICENTIA11K, class_type = data_args['args']['class'], unfold = data_args['args']['unfold'], 
                                                    spectrogram_control = data_args['args']['spectrogram'], signal_type = data_args['args']['signal_type'], ecg_crop_lengths = crop,
                                                    ecg_sequence_length = ecg_length),
                            batch_size=max(1, data_args['args']['batch_size'] // 50), num_workers=min(max(data_args['args']['batch_size'] // 50, 4), 20),
                            pin_memory=True, drop_last=False, shuffle=True, collate_fn=icentia11k_dataset_collate_fn                           
                        ),
                        Mode.eval: data.DataLoader(
                            IcentiaLoader_MLP_batch(path,split = VALIDATION_SPLIT_ICENTIA11K, class_type = data_args['args']['class'], unfold = data_args['args']['unfold'], 
                                                    spectrogram_control = data_args['args']['spectrogram'], signal_type = data_args['args']['signal_type'], ecg_crop_lengths = crop,
                                                    ecg_sequence_length = ecg_length),
                            batch_size=max(1, data_args['args']['batch_size'] // 50), num_workers=min(max(data_args['args']['batch_size'] // 50, 4), 20),
                            pin_memory=True, drop_last=False, shuffle=True, collate_fn=icentia11k_dataset_collate_fn                   
                        )
                    }
        
        elif data_args['type'] == "CinC":
                print("DataSet : PhysioNet/CinC")
                
                
                
                ecg_leads, ecg_labels, fs, ecg_names = load_references(path)
         
                
                training_split = TRAINING_SPLIT_CHALLANGE
                validation_split = VALIDATION_SPLIT_CHALLANGE
                
                  
                return {
                        Mode.train: data.DataLoader(
                            CinCLoader(ecg_leads=[ecg_leads[index] for index in training_split], class_type = data_args['args']['class'],
                                ecg_labels=[ecg_labels[index] for index in training_split], fs=fs,
                                augmentation_pipeline=None if data_args['args']['aug'] else AugmentationPipeline(
                                    AUGMENTATION_PIPELINE_CONFIG_2C)),
                            batch_size=data_args['args']['batch_size'], num_workers=data_args['args']['num_workers'], pin_memory=True,
                            drop_last=False, shuffle=True),
                        Mode.eval: data.DataLoader(
                            CinCLoader(ecg_leads=[ecg_leads[index] for index in validation_split],class_type = data_args['args']['class'],
                                ecg_labels=[ecg_labels[index] for index in validation_split], fs=fs,
                                augmentation_pipeline=None),
                            batch_size=data_args['args']['batch_size'], num_workers=data_args['args']['num_workers'], pin_memory=True,
                            drop_last=False, shuffle=False)
                    }
                
                          

def load_references(folder: str = '../training') -> Tuple[List[np.ndarray], List[str], int, List[str]]:
    """
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.
    Returns
    -------
    ecg_leads : List[np.ndarray]
        EKG Signale.
    ecg_labels : List[str]
        Gleiche Laenge wie ecg_leads. Werte: 'N','A','O','~'
    fs : int
        Sampling Frequenz.
    ecg_names : List[str]
        Name der geladenen Dateien
    """
    # Check Parameter
    assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
    assert os.path.exists(folder), 'Parameter folder existiert nicht!'
    # Initialisiere Listen für leads, labels und names
    ecg_leads: List[np.ndarray] = []
    ecg_labels: List[str] = []
    ecg_names: List[str] = []
    # Setze sampling Frequenz
    fs: int = 300
    # Lade references Datei
    with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # Iteriere über jede Zeile
        for row in csv_reader:
            # Lade MatLab Datei mit EKG lead and label
            data = sio.loadmat(os.path.join(folder, row[0] + '.mat'))
            ecg_leads.append(data['val'][0])
            ecg_labels.append(row[1])
            ecg_names.append(row[0])
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ecg_leads)))
    return ecg_leads, ecg_labels, fs, ecg_names

               

def icentia11k_dataset_collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for PyTorch DataLoader
    :param inputs: (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]])) Batch of data
    :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Packed data
    """
    # Pack data
    ecg_leads = torch.cat([input[0] for input in inputs], dim=0)

    labels = torch.cat([input[1] for input in inputs], dim=0)
    return ecg_leads,  labels  

           
# def get_path(args):
#     path = {
#         Mode.train: args['path']["train"],
#         Mode.eval: args['path']["eval"]
#     }
#     return path
    
def get_trans():
    transforms = {
        Mode.train: Compose([RandomNoise(0.05, 0.5), RandomShift(10, 0.5)]),
        Mode.eval: lambda x: x
    }
    return transforms


d
    

    
