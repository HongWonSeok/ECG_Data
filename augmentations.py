from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from scipy import signal


#
class Compose:

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    # 객체가 함수처럼 호출될 때 동작하는 메소드 
    # transform에 저장된 변환을 순차적으로 적용.
    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal


class RandomShift:

    # max_num_sample을 통해 신호가 이동할 수 있는 최대 샘플 수 정의
    # probability를 통해 실제로 shift가 적용될 확률 정의. (0~1의 값으로 default는 0.5)
    # 이 두개를 파라미터로 받아와 _m과 _p에 적용
    def __init__(self, max_num_samples: int, probability=0.5) -> None:
        self._m = max_num_samples
        self._p = probability

    # 매개변수로 ECG신호인 signal을 받음
    def __call__(self, signal):
        
        # torch.rand(1)을 통해 0~1 사이의 수를 하나 생성하고, 이것이 probability보다 큰지 비교
        # 즉, 만약 probability가 0.5면 즉, 0.5보다 크면 원본신호 그대로 return 즉,0.5 확률로 shift할건지 결정.
        if torch.rand(1) > self._p:
            return signal

        # shift하기로 됬으면 진행
        # -m과 m사이의 random 값을 생성하여 이동시킬 샘플 수를 정의. 그런다음 torch.roll을 통해 signal을 이동 시키고 신호를 return시킴.
        # roll()의 세번째 인자 0은 신호를 어떤 차원에서 이동시킬 지 정의 / 여기서는 0번 차원 즉 첫번쨰 차원을 이동시킴.
        # 신호는 dataloader에서 보면 1개의 index로 가져와서 1차원이기 때문에 왼쪽이나 오른쪽으로 이동하게됨.
        return torch.roll(signal, torch.randint(-self._m, self._m, (1, )).item(), 0)


class RandomNoise:

    # 이거도 마찬가지로 max_amplitude와 probability를 매개변수로 받음.
    # probability는 위에와 동일하고, max_amplitude또한 위의 max_num_samples와 비슷하게 얼마나 noise를 줄건지 noise의 최대 진폭을 정의함.
    def __init__(self, max_amplitude, probability) -> None:
        self._a = max_amplitude
        self._p = probability

    # p에 따라 확률적으로 Random noise진행.
    def __call__(self, signal):
        if torch.rand(1) > self._p:
            return signal

        # noise를 주기로 했을 때 return
        # torch.rand(len(signal))을 통해 signal의 길이만큼의 random 값(0~1)을 생성 하여 -0.5 한 값에 2를 곱하고 _a를 곱합.
        # 이 값이 원본 signal값과 더해져서 noise가 추가됨.
        return signal + (torch.rand(len(signal)) - 0.5) * 2. * self._a


class AugmentationPipeline(nn.Module):
    """
    This class implements an augmentation pipeline for ecg leads.
    Inspired by: https://arxiv.org/pdf/2009.04398.pdf
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor method
        :param config: (Dict[str, Any]) Config dict
        """
        # Call super constructor
        super(AugmentationPipeline, self).__init__()
        # Save parameters
        self.ecg_sequence_length: int = config["ecg_sequence_length"]
        self.p_scale: float = config["p_scale"]
        self.p_drop: float = config["p_drop"]
        self.p_cutout: float = config["p_cutout"]
        self.p_shift: float = config["p_shift"]
        self.p_resample: float = config["p_resample"]
        self.p_random_resample: float = config["p_random_resample"]
        self.p_sine: float = config["p_sine"]
        self.p_band_pass_filter: float = config["p_band_pass_filter"]
        self.fs: int = config["fs"]
        self.scale_range: Tuple[float, float] = config["scale_range"]
        self.drop_rate = config["drop_rate"]
        self.interval_length: float = config["interval_length"]
        self.max_shift: int = config["max_shift"]
        self.resample_factors: Tuple[float, float] = config["resample_factors"]
        self.max_offset: float = config["max_offset"]
        self.resampling_points: int = config["resampling_points"]
        self.max_sine_magnitude: float = config["max_sine_magnitude"]
        self.sine_frequency_range: Tuple[float, float] = config["sine_frequency_range"]
        self.kernel: Tuple[float, ...] = config["kernel"]
        self.fs: int = config["fs"]
        self.frequencies: Tuple[float, float] = config["frequencies"]

    def scale(self, ecg_lead: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """
        Scale augmentation:  Randomly scaling data
        :param ecg_lead: (torch.Tensor) ECG leads
        :param scale_range: (Tuple[float, float]) Min and max scaling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Get random scalar
        random_scalar = torch.from_numpy(np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)).float()
        # Apply scaling
        ecg_lead = random_scalar * ecg_lead
        return ecg_lead

    def drop(self, ecg_lead: torch.Tensor, drop_rate: float = 0.025) -> torch.Tensor:
        """
        Drop augmentation: Randomly missing signal values
        :param ecg_lead: (torch.Tensor) ECG leads
        :param drop_rate: (float) Relative number of samples to be dropped
        :return: (torch.Tensor) ECG lead augmented
        """
        # Estimate number of sample to be dropped
        num_dropped_samples = int(ecg_lead.shape[-1] * drop_rate)
        # Randomly drop samples
        ecg_lead[..., torch.randperm(ecg_lead.shape[-1])[:max(1, num_dropped_samples)]] = 0.
        return ecg_lead

    def cutout(self, ecg_lead: torch.Tensor, interval_length: float = 0.1) -> torch.Tensor:
        """
        Cutout augmentation: Set a random interval signal to 0
        :param ecg_lead: (torch.Tensor) ECG leads
        :param interval_length: (float) Interval lenght to be cut out
        :return: (torch.Tensor) ECG lead augmented
        """
        # Estimate interval size
        interval_size = int(ecg_lead.shape[-1] * interval_length)
        # Get random starting index
        index_start = torch.randint(low=0, high=ecg_lead.shape[-1] - interval_size, size=(1,))
        # Apply cut out
        ecg_lead[index_start:index_start + interval_size] = 0.
        return ecg_lead

    def shift(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000, max_shift: int = 4000) -> torch.Tensor:
        """
        Shift augmentation: Shifts the signal at random
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_shift: (int) Max applied shift
        :return: (torch.Tensor) ECG lead augmented
        """
        # Generate shift
        shift = torch.randint(low=0, high=max_shift, size=(1,))
        # Apply shift
        ecg_lead = torch.cat([torch.zeros_like(ecg_lead)[..., :shift], ecg_lead], dim=-1)[:ecg_sequence_length]
        return ecg_lead

    def resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                 resample_factors: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Resample augmentation: Resamples the ecg lead
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param resample_factor: (Tuple[float, float]) Min and max value for resampling
        :return: (torch.Tensor) ECG lead augmented
        """
        # Generate resampling factor
        resample_factor = torch.from_numpy(
            np.random.uniform(low=resample_factors[0], high=resample_factors[1], size=1)).float()
        # Resample ecg lead
        ecg_lead = F.interpolate(ecg_lead[None, None], size=int(resample_factor * ecg_lead.shape[-1]), mode="linear",
                                 align_corners=False)[0, 0]
        # Apply max length if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def random_resample(self, ecg_lead: torch.Tensor, ecg_sequence_length: int = 18000,
                        max_offset: float = 0.03, resampling_points: int = 4) -> torch.Tensor:
        """
        Random resample augmentation: Randomly resamples the signal
        :param ecg_lead: (torch.Tensor) ECG leads
        :param ecg_sequence_length: (int) Fixed max length of sequence
        :param max_offset: (float) Max resampling offsets between 0 and 1
        :param resampling_points: (int) Initial resampling points
        :return: (torch.Tensor) ECG lead augmented
        """
        # Make coordinates for resampling
        coordinates = 2. * (torch.arange(ecg_lead.shape[-1]).float() / (ecg_lead.shape[-1] - 1)) - 1
        # Make offsets
        offsets = F.interpolate(((2 * torch.rand(resampling_points) - 1) * max_offset)[None, None],
                                size=ecg_lead.shape[-1], mode="linear", align_corners=False)[0, 0]
        # Make grid
        grid = torch.stack([coordinates + offsets, coordinates], dim=-1)[None, None].clamp(min=-1, max=1)
        # Apply resampling
        ecg_lead = F.grid_sample(ecg_lead[None, None, None], grid=grid, mode='bilinear', align_corners=False)[0, 0, 0]
        # Apply max lenght if needed
        ecg_lead = ecg_lead[:ecg_sequence_length]
        return ecg_lead

    def sine(self, ecg_lead: torch.Tensor, max_sine_magnitude: float = 0.2,
             sine_frequency_range: Tuple[float, float] = (0.2, 1.), fs: int = 300) -> torch.Tensor:
        """
        Sine augmentation: Add a sine wave to the entire sample
        :param ecg_lead: (torch.Tensor) ECG leads
        :param max_sine_magnitude: (float) Max magnitude of sine to be added
        :param sine_frequency_range: (Tuple[float, float]) Sine frequency rand
        :param fs: (int) Sampling frequency
        :return: (torch.Tensor) ECG lead augmented
        """
        # Get sine magnitude
        sine_magnitude = torch.from_numpy(np.random.uniform(low=0, high=max_sine_magnitude, size=1)).float()
        # Get sine frequency
        sine_frequency = torch.from_numpy(
            np.random.uniform(low=sine_frequency_range[0], high=sine_frequency_range[1], size=1)).float()
        # Make t vector
        t = torch.arange(ecg_lead.shape[-1]) / float(fs)
        # Make sine vector
        sine = torch.sin(2 * math.pi * sine_frequency * t + torch.rand(1)) * sine_magnitude
        # Apply sine
        ecg_lead = sine + ecg_lead
        return ecg_lead

    def band_pass_filter(self, ecg_lead: torch.Tensor, frequencies: Tuple[float, float] = (0.2, 45.),
                         fs: int = 300) -> torch.Tensor:
        """
        Low pass filter: Applies a band pass filter
        :param ecg_lead: (torch.Tensor) ECG leads
        :param frequencies: (Tuple[float, float]) Frequencies of the band pass filter
        :param fs: (int) Sample frequency
        :return: (torch.Tensor) ECG lead augmented
        """
        # Init filter
        sos = signal.butter(10, frequencies, 'bandpass', fs=fs, output='sos')
        ecg_lead = torch.from_numpy(signal.sosfilt(sos, ecg_lead.numpy()))
        return ecg_lead

    def forward(self, ecg_lead: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies augmentation to input tensor
        :param ecg_lead: (torch.Tensor) ECG leads
        :return: (torch.Tensor) ECG lead augmented
        """
        # Apply cut out augmentation
        if random.random() <= self.p_cutout:
            ecg_lead = self.cutout(ecg_lead, interval_length=self.interval_length)
        # Apply drop augmentation
        if random.random() <= self.p_drop:
            ecg_lead = self.drop(ecg_lead, drop_rate=self.drop_rate)
        # Apply random resample augmentation
        if random.random() <= self.p_random_resample:
            ecg_lead = self.random_resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
                                            max_offset=self.max_offset, resampling_points=self.resampling_points)
        # Apply resample augmentation
        if random.random() <= self.p_resample:
            ecg_lead = self.resample(ecg_lead, ecg_sequence_length=self.ecg_sequence_length,
                                     resample_factors=self.resample_factors)
        # Apply scale augmentation
        if random.random() <= self.p_scale:
            ecg_lead = self.scale(ecg_lead, scale_range=self.scale_range)
        # Apply shift augmentation
        if random.random() <= self.p_shift:
            ecg_lead = self.shift(ecg_lead, ecg_sequence_length=self.ecg_sequence_length, max_shift=self.max_shift)
        # Apply sine augmentation
        if random.random() <= self.p_sine:
            ecg_lead = self.sine(ecg_lead, max_sine_magnitude=self.max_sine_magnitude,
                                 sine_frequency_range=self.sine_frequency_range, fs=self.fs)
        # Apply low pass filter
        if random.random() <= self.p_band_pass_filter:
            ecg_lead = self.band_pass_filter(ecg_lead, frequencies=self.frequencies, fs=self.fs)
        return ecg_lead