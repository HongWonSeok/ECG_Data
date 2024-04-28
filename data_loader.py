from typing import Callable, Tuple, List, Optional

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import os
import torch.utils.data as data
from torchaudio.transforms import Spectrogram
from bisect import bisect
import numpy as np
import gzip
import pickle



class MITLoader_MLP(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal, torch.tensor(label).long()
    
    
    
class MITLoader_CNN_Transformer(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal.unsqueeze(0), torch.tensor(label).long()


class Binary_MITLoader_MLP(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])

        if label != 0:
            label = 1
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal, torch.tensor(label).long()
    
    
    
class Binary_MITLoader_CNN_Transformer(data.Dataset):

    def __init__(self, csv_file, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        self.annotations = pd.read_csv(csv_file).values
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, item):
        signal = self.annotations[item, :-1]
        label = int(self.annotations[item, -1])
        
        if label != 0:
            label = 1
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal.unsqueeze(0), torch.tensor(label).long()




class IcentiaLoader_MLP(data.Dataset):

    def __init__(self,path, transforms: Callable = lambda x: x) -> None:
        super().__init__()
        
        self.root_dir = path
        self.signal_paths = self.get_signal_path()
        self.transforms = transforms

    def get_signal_path(self):
        signal_path = glob.glob(os.path.join(self.root_dir,'**','*.npy'),recursive=True)
        print('siganl_done')
        return signal_path
    
    def __len__(self):
        return len(self.signal_paths)
    

    def __getitem__(self, i):
        
        data = np.load(self.signal_paths[i], allow_pickle=True)
        signal = data.item().get('signal')
        label = data.item().get('label')
        
        if label == 'N':
            label = 0
        else :
            label = 1
        # TODO: add augmentations
        signal = torch.from_numpy(signal).float()
        signal = self.transforms(signal)

        return signal, torch.tensor(label).long()
    
class IcentiaLoader_MLP_batch(data.Dataset):

    def __init__(self, path: str, split: List[int], class_type : str, signal_type :str,
                 ecg_crop_lengths: Tuple[int, int] = (9000, 18000), unfold : bool = False, spectrogram_control : bool = False,
                 original_fs: int = 250, spectrogram_length: int = 563, ecg_sequence_length: int = 18000,
                 ecg_window_size: int = 256, ecg_step: int = 256 - 32, normalize: bool = True, fs: int = 300,
                 spectrogram_n_fft: int = 64, spectrogram_win_length: int = 64, spectrogram_power: int = 1,
                 spectrogram_normalized: bool = True, random_seed: Optional[int] = 1904) -> None:
        super().__init__()
        print('IcentiaLoader_MLP_batch')
        self.path = path
        self.split = split
        self.ecg_crop_lengths = ecg_crop_lengths
        self.class_type = class_type
        self.signal_type = signal_type
        self.unfold = unfold
        self.spectrogram_control = spectrogram_control
        self.spectrogram_length = spectrogram_length
        self.original_fs = original_fs
        self.ecg_sequence_length = ecg_sequence_length
        self.ecg_window_size = ecg_window_size
        self.ecg_step = ecg_step
        self.normalize = normalize
        self.fs = fs
        self.random_seed = random_seed
        self.spectrogram_module = Spectrogram(n_fft=spectrogram_n_fft, win_length=spectrogram_win_length,
                                              hop_length=spectrogram_win_length // 2, power=spectrogram_power,
                                              normalized=spectrogram_normalized)
        
        self.paths: List[Tuple[str, str]] = []
        for index in self.split:
            self.paths.append((os.path.join(self.path, "{}_batched.pkl.gz".format(str(index).zfill(5))),
                               os.path.join(self.path, "{}_batched_lbls.pkl.gz".format(str(index).zfill(5)))))
        # Check if files exists
        for file_input, file_label in self.paths:
            assert os.path.isfile(file_input), "File {} not found!".format(file_input)
            assert os.path.isfile(file_label), "File {} not found!".format(file_label)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch단위로 파일 가져옴(돌려보니까 그냥 sample가져옴).
        with gzip.open(self.paths[item][0], "rb") as file:
            inputs = torch.from_numpy(pickle.load(file)).float()
        with gzip.open(self.paths[item][1], "rb") as file:
            labels = pickle.load(file)
        # Use generator if random seed is utilized
        if self.random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_seed)
        else:
            generator = None
        # Make crop indexes
        crop_indexes_low = torch.randint(
            low=0,
            high=int(inputs.shape[-1] - (self.fs / self.original_fs) * max(self.ecg_crop_lengths)),
            size=(inputs.shape[0],), generator=generator)
        crop_indexes_length = torch.randint(
            low=int((self.fs / self.original_fs) * min(self.ecg_crop_lengths)),
            high=int((self.fs / self.original_fs) * max(self.ecg_crop_lengths)),
            size=(inputs.shape[0],), generator=generator)
        # Crop signals
        inputs = [input[low:low + length] for input, low, length in zip(inputs, crop_indexes_low, crop_indexes_length)]
        # Interpolate signals
        
        inputs = [F.interpolate(input[None, None], scale_factor=self.fs / self.original_fs, mode="linear",
                                align_corners=False)[0, 0] for input in inputs]
        # Normalize signals
        if self.normalize:
            inputs = [(input - input.mean()) / (input.std() + 1e-08) for input in inputs]
    
            
        if self.spectrogram_control:
            spectrograms = [self.spectrogram_module(input).abs().clamp(min=1e-08).log() for input in inputs]
        
        # Pad inputs
        inputs = torch.stack(
            [F.pad(input, pad=(0, self.ecg_sequence_length - input.shape[0]), value=0., mode="constant")
             for input in inputs], dim=0)
             
        if self.unfold:
            inputs = inputs.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)
            
        if self.spectrogram_control:
            spectrograms = torch.stack(
            [F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]), value=0.,
                   mode="constant").permute(1, 0) for spectrogram in spectrograms], dim=0)

      
        if self.signal_type == 'rythm':
            # Get labels
            # classes에 class저장해서 return 할거임.
            classes = []
            for index, label in enumerate(labels):
                # rhythm type의 라벨 들 찾음.
                label = label["rtype"]
                # Make class tensor
                noise = 0
                normal = 0
                af = 0
                afl = 0
                index_low = crop_indexes_low[index].item()
                index_high = (crop_indexes_low[index].item() + crop_indexes_length[index].item())
                for class_index, rtype in enumerate(label):
                    rtype = torch.from_numpy(rtype)
                    if rtype.shape != (0,):
                        if class_index in (0, 1, 2):
                            noise += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        elif class_index == 3:
                            normal += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        elif class_index == 4:
                            af += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        else:
                            afl += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                # If AFL is present label is set to AFL, if AF is present and not AFL then labels is set to AF
                # binary
                if self.class_type == 'binary':
                    if afl > 0 or af > 0 or noise > 0:
                        classes.append(1)
                    else:
                        classes.append(0) 
                        
                elif self.class_type == 'multi':
                    if afl > 0:
                        classes.append(3)
                    elif af > 0:
                        classes.append(2)
                    # If there is more noise in the data labels is set to noise and no AFL or AF is present
                    elif noise > 0:
                        classes.append(0)
                    # If no noise, AFL or AF normal class is used
                    else:
                        classes.append(1)    
                
            classes = torch.tensor(classes, dtype=torch.long)
            # Classes to one hot
            # one_hot_classes = F.one_hot(classes, num_classes=4) 
        
        if self.signal_type == 'beat':
            # Get labels
            # classes에 class저장해서 return 할거임.
            classes = []
            for index, label in enumerate(labels):
                # rhythm type의 라벨 들 찾음.
                label = label["btype"]
                # Make class tensor
                N = 0
                Q = 0
                V = 0
                S = 0
                index_low = crop_indexes_low[index].item()
                index_high = (crop_indexes_low[index].item() + crop_indexes_length[index].item())
                for class_index, rtype in enumerate(label):
                    rtype = torch.from_numpy(rtype)
                    if rtype.shape != (0,):
                        if class_index == 0:
                            Q += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        elif class_index == 1:
                            N += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        elif class_index == 2:
                            S += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                        elif class_index == 4:
                            V += torch.logical_and(rtype >= index_low, rtype <= index_high).sum().item()
                # If AFL is present label is set to AFL, if AF is present and not AFL then labels is set to AF
                # binary
                if self.class_type == 'binary':
                    if V > 0 or S > 0 or Q > 0:
                        classes.append(1)
                    else:
                        classes.append(0) 
                        
                elif self.class_type == 'multi':
                    if V > 0:
                        classes.append(1)
                    elif S > 0:
                        classes.append(2)
                    # If there is more noise in the data labels is set to noise and no AFL or AF is present
                    elif Q > 0:
                        classes.append(3)
                    # If no noise, AFL or AF normal class is used
                    else:
                        classes.append(0)    
                
            classes = torch.tensor(classes, dtype=torch.long)
            # Classes to one hot
            # one_hot_classes = F.one_hot(classes, num_classes=4) 
            
        return inputs,  classes
    
            
            
class CinCLoader(data.Dataset):
    """
    This class implements the ECG dataset for atrial fibrillation classification.
    """

    def __init__(self, ecg_leads: List[np.ndarray], ecg_labels: List[str], class_type : str,
                 augmentation_pipeline: Optional[nn.Module] = None, spectrogram_length: int = 563, unfold : bool = False, spectrogram_control : bool = False,
                 ecg_sequence_length: int = 18000, ecg_window_size: int = 256, ecg_step: int = 256 - 32,
                 normalize: bool = True, fs: int = 300, spectrogram_n_fft: int = 64, spectrogram_win_length: int = 64,
                 spectrogram_power: int = 1, spectrogram_normalized: bool = True, two_classes: bool = False) -> None:
        """
        Constructor method
        :param ecg_leads: (List[np.ndarray]) ECG data as list of numpy arrays
        :param ecg_labels: (List[str]) ECG labels as list of strings (N, O, A, ~)
        :param augmentation_pipeline: (Optional[nn.Module]) Augmentation pipeline
        :param spectrogram_length: (int) Fixed spectrogram length (achieved by zero padding)
        :param spectrogram_shape: (Tuple[int, int]) Final size of the spectrogram
        :param ecg_sequence_length: (int) Fixed length of sequence
        :param ecg_window_size: (int) Window size to be applied during unfolding
        :param ecg_step: (int) Step size of unfolding
        :param normalize: (bool) If true signal is normalized to a mean and std of zero and one respectively
        :param fs: (int) Sampling frequency
        :param spectrogram_n_fft: (int) FFT size utilized in spectrogram
        :param spectrogram_win_length: (int) Spectrogram window length
        :param spectrogram_power: (int) Power utilized in spectrogram
        :param spectrogram_normalized: (int) If true spectrogram is normalized
        :param two_classes: (bool) If true only two classes are utilized
        """
        # Call super constructor
        super(CinCLoader, self).__init__()
        # Save parameters
        self.ecg_leads: List[torch.Tensor] = [torch.from_numpy(data_sample).float() for data_sample in ecg_leads]
        self.augmentation_pipeline: nn.Module = augmentation_pipeline \
            if augmentation_pipeline is not None else nn.Identity()
        self.class_type = class_type
        self.spectrogram_length: int = spectrogram_length
        self.unfold = unfold
        self.spectrogram_control = spectrogram_control
        self.ecg_sequence_length: int = ecg_sequence_length
        self.ecg_window_size: int = ecg_window_size
        self.ecg_step: int = ecg_step
        self.normalize: bool = normalize
        self.fs: int = fs
     
        # Make labels
        self.ecg_labels: List[torch.Tensor] = []
        if self.class_type == 'binary':
            ecg_leads_: List[torch.Tensor] = []
            for index, ecg_label in enumerate(ecg_labels):
                if ecg_label == "N":
                    self.ecg_labels.append(0)
                    ecg_leads_.append(self.ecg_leads[index])
                else:
                    self.ecg_labels.append(1)
                    ecg_leads_.append(self.ecg_leads[index])
            self.ecg_leads = ecg_leads_
        if self.class_type == 'multi':
            for ecg_label in ecg_labels:
                if ecg_label == "N":
                    self.ecg_labels.append(0)
                elif ecg_label == "O":
                    self.ecg_labels.append(1)
                elif ecg_label == "A":
                    self.ecg_labels.append(2)
                elif ecg_label == "~":
                    self.ecg_labels.append(3)
                else:
                    raise RuntimeError("Invalid label value detected!")
        # Make spectrogram module
        
        if self.spectrogram_control:
            self.spectrogram_module: nn.Module = Spectrogram(n_fft=spectrogram_n_fft, win_length=spectrogram_win_length,
                                                            hop_length=spectrogram_win_length // 2,
                                                            power=spectrogram_power, normalized=spectrogram_normalized)

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.ecg_leads)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single instance of the dataset
        :param item: (int) Index of the dataset instance to be returned
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) ECG lead, spectrogram, label
        """
        # Get ecg lead, label, and name
        
        
        ecg_lead = self.ecg_leads[item][:self.ecg_sequence_length]
  
        
        ecg_label = self.ecg_labels[item]
        # Apply augmentations
        ecg_lead = self.augmentation_pipeline(ecg_lead)
        # Normalize signal if utilized
        if self.normalize:
            ecg_lead = (ecg_lead - ecg_lead.mean()) / (ecg_lead.std() + 1e-08)
        # Compute spectrogram of ecg_lead
        
        if self.spectrogram_control:
            spectrogram = self.spectrogram_module(ecg_lead)
            spectrogram = torch.log(spectrogram.abs().clamp(min=1e-08))
            # Pad spectrogram to the desired shape
            spectrogram = F.pad(spectrogram, pad=(0, self.spectrogram_length - spectrogram.shape[-1]),
                                value=0., mode="constant").permute(1, 0)
        # Pad ecg lead
        ecg_lead = F.pad(ecg_lead, pad=(0, self.ecg_sequence_length - ecg_lead.shape[0]), value=0., mode="constant")
        # Unfold ecg lead
        if self.unfold:
            ecg_lead = ecg_lead.unfold(dimension=-1, size=self.ecg_window_size, step=self.ecg_step)
      
        ecg_lead = torch.tensor(ecg_lead, dtype=torch.float32)
        ecg_label = torch.tensor(ecg_label, dtype=torch.long)
        return ecg_lead , ecg_label