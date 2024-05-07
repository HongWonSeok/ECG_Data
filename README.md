# ECG_Data

The data loader and data refer to the study `ECG-DualNet: Atrial Fibrillation Classification in Electrocardiography using Deep Learning.`

## Data 

### Icentia
- This data is used for pre-training.

* Data
  * The source of this data is [Icentia](https://studtudarmstadtde-my.sharepoint.com/personal/christoph_reich_stud_tu-darmstadt_de/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchristoph%5Freich%5Fstud%5Ftu%2Ddarmstadt%5Fde%2FDocuments%2FUni%2FECG%5FClassification%2Fdata%2Ezip&parent=%2Fpersonal%2Fchristoph%5Freich%5Fstud%5Ftu%2Ddarmstadt%5Fde%2FDocuments%2FUni%2FECG%5FClassification&ga=1)
  * This data is identical to the [PhysioNet Icentia](https://www.physionet.org/content/icentia11k-continuous-ecg/1.0/) dataset from PhysioNet. It is a dataset where the labels at the R peaks of the original data have been organized into a list.
  * The list of labels contains the positions of the R peaks in the ECG signal for the same disease, stored in each row.

  > ex) For example, the positions of the R peaks related to disease Q in row 0 are stored as [26334, 26434, 26551, 26758, 26946, 27139, 27326, ...], and in row 1, the positions of the R peaks related to disease N are stored as [8, 178, 354, ..., 1048192, 1048333, 1048548].

  * The labels for the beats in each row are as follows. `0 -> Q` , `1 -> N`, `2 -> S`, `4 -> V`
  * The labels for the rythm in each row are as follows. `0,1,2 -> noise`, `3 -> normal`, `4 -> af`, `5 -> afl`
  
* Dataloader
  * One sample is extracted per segment, resulting in approximately 50 samples per person.
  * A sample is extracted from a random position of a specified length, and the final label is determined by comparing the R peaks in the extracted signal with the label list.
  * In this way, about 50 samples are drawn for one person in the get_item() function.
  * Training is conducted for 10,000 individuals, and testing is conducted for the remaining 1,000 individuals.

------
### PhysioNet/CinC2017
* Data
  * This data is intended for rhythm-based analysis.
  * The training and testing datasets are classified according to lists stored in `CinC_split.py.`
  * The data was sourced from the [CinC](https://physionet.org/content/challenge-2017/1.0.0/) dataset on PhysioNet.
  * The training2017 dataset was used for this data.
    
* Dataloader
  * data loader was constructed using classes and functions from `util.py`, and `augmentations.py`, following the methodology described in ECG-DualNet.  
  * The `load_references()` function in `util.py` is used to retrieve the ECG signals and the corresponding list of labels.
  * For each ECG data, it is collected up to a length of 18,000. If the length does not reach 18,000, padding is applied.

------
### MIT-BIH
- The MIT-BIH code has also been included.


## How to use

1. Icentia
```
python train.py --config ./configs/config_Icentia_batch.json
```

2. PhysioNet/CinC2017
```
python train.py --config ./configs/config_CinC.json
```

