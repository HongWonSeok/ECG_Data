from typing import Tuple, List, Optional

from model.models import SIMPLEMLP_MIT, SIMPLEMLP_CinC, SIMPLEMLP_Icentia
from model.metrics import Metrics
from model.loss import Weighted_CE_Loss
from data.data_loader import MITLoader_MLP, IcentiaLoader_MLP_batch, CinCLoader
import torch.utils.data as data
from data.augmentations import  Compose, RandomNoise, RandomShift, AugmentationPipeline, AUGMENTATION_PIPELINE_CONFIG_2C
from configs.CinC_split import TRAINING_SPLIT_CHALLANGE, VALIDATION_SPLIT_CHALLANGE

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
        
      
            
    if data_args['args']['class'] == 'binary':  
        
        if data_args['type'] == "MIT-BIH":        
            print("DataSet : MIT-BIH")
            
            path = get_path(data_args['args'])
            transforms = get_trans()
            
            if model_args['type'] == 'MLP':
                return {
                    Mode.train: data.DataLoader(
                        MITLoader_MLP(path[Mode.train], transforms[Mode.train]),
                        batch_size=data_args['args']['batch_size'],
                        num_workers=data_args['args']['num_workers'],
                        shuffle=True,
                        drop_last=True
                    ),
                    Mode.eval: data.DataLoader(
                        MITLoader_MLP(path[Mode.eval], transforms[Mode.eval]),
                        batch_size=data_args['args']['batch_size'],
                        num_workers=data_args['args']['num_workers'],
                        shuffle=False,
                        drop_last=True
                    )
                }

        
                
        if data_args['type'] == "Icentia_batch":            
            print("DataSet : Icentia_batch")
            
            path = data_args['args']['path']

            # 총 11000명에 대해 train 10000명 valid 1000명
            TRAINING_SPLIT_ICENTIA11K = list(range(10000))
            VALIDATION_SPLIT_ICENTIA11K = list(range(10000,11000))

            # beat와 rythm에 따라 각 segment를 랜덤으로 자르는 길이를 다르게함.
            if data_args['args']['signal_type'] == 'beat':
                crop = (250,500)
                ecg_length = 500
            elif data_args['args']['signal_type'] == 'rythm':
                crop = (9000,18000)
                ecg_length = 18000
            
            if model_args['type'] == 'MLP':
                '''
                class type : binary
                unfold : unfod여부 
                spectrogram_control : spectrogram 변환 여부, dataloder에는 구현되어있지만, train epoch코드에는 아직 미구현 필요시 추가하면됨.
                signal type : beat
                ecg_crop_length : segment하나당 어느정도의 길이로 자를 것인지에 관한 것, crop
                ecg_sequence_length : 총 padding되는 길이, beat는 500, rythm은 18000으로 설정.
                '''
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
            
            path = data_args['args']['path']
            
            print("DataSet : PhysioNet/CinC")

            # load_references()를 통해 signal과 label를 받아와서 사용
            ecg_leads, ecg_labels, fs, ecg_names = load_references(path)
        

            # configs/CinC_split.py에 작성되어있음.
            # Icentia에서의 official은 아니고, README에 작성된 논문에서 이렇게 사용하여서 그대로 사용함.
            training_split = TRAINING_SPLIT_CHALLANGE
            validation_split = VALIDATION_SPLIT_CHALLANGE

            '''
            AugmentationPipeline : data/augmentations.py에 작성되어있음.
            AUGMENTATION_PIPELINE_CONFIG_2C : data/augmentations.py에 작성되어있음.
            '''
                
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
                

def get_path(args):
    path = {
        Mode.train: args['path']["train"],
        Mode.eval: args['path']["eval"]
    }
    return path
    
def get_trans():
    transforms = {
        Mode.train: Compose([RandomNoise(0.05, 0.5), RandomShift(10, 0.5)]),
        Mode.eval: lambda x: x
    }
    return transforms
                    

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

               


def icentia11k_dataset_collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for PyTorch DataLoader
    :param inputs: (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]])) Batch of data
    :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Packed data
    """
    # Pack data
    ecg_leads = torch.cat([input[0] for input in inputs], dim=0)
    spectrograms = torch.cat([input[1] for input in inputs], dim=0)
    labels = torch.cat([input[2] for input in inputs], dim=0)
    return ecg_leads, spectrograms, labels

    



def get_model(args, data_args):
    name = args['type'].upper()
    
    if name == "MLP":
        
        if data_args["type"] == "MIT-BIH":
            return SIMPLEMLP_MIT(args['args']['num_classes'])
        
        elif data_args["type"] == 'CinC':  
            return SIMPLEMLP_CinC(args['args']['num_classes'])

        elif data_args["type"] == 'Icentia_batch':  
            return SIMPLEMLP_Icentia(args['args']['num_classes'])


 
def get_metric(args, model_args, metrics):
   
    if args['type'] == "f1":
        return metrics.f1_score(args, model_args)

    elif args['type'] == "AUROC":
        return metrics.AUROC(args, model_args)

    elif args['type'] == "AUPRC":
        return metrics.AUPRC(args, model_args)
    

def get_loss(args):
    
    if args['type'] == "Weighted_CE_loss":

        return Weighted_CE_Loss(torch.tensor(args['weight']))
    

    
