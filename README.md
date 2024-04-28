# ECG_Data

데이터 로더 및 데이터는 `ECG-DualNet: Atrial Fibrillation Classification in Electrocardiography using Deep Learning`을 참고했습니다.

## Data 

1. Icentia
* 데이터
  * Icentia 데이터는 [Icentia](https://studtudarmstadtde-my.sharepoint.com/personal/christoph_reich_stud_tu-darmstadt_de/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchristoph%5Freich%5Fstud%5Ftu%2Ddarmstadt%5Fde%2FDocuments%2FUni%2FECG%5FClassification%2Fdata%2Ezip&parent=%2Fpersonal%2Fchristoph%5Freich%5Fstud%5Ftu%2Ddarmstadt%5Fde%2FDocuments%2FUni%2FECG%5FClassification&ga=1)에서 가져왔습니다.
  * 기존에 말씀드린 physionet에서의 데이터셋인 [PhysioNet Icentia](https://www.physionet.org/content/icentia11k-continuous-ecg/1.0/)와 동일한 데이터이며, 기존데이터의 R피크에서의 label을 list로 정리해둔 데이터셋입니다.
  * label에 대한 list는 각 행에 같은 질병에 대해 ECG신호상의 R 피크의 위치가 저장되어있습니다. <br/>
  > ex) 예를들어 0번행의 경우 Q에 관한 R피크의 위치가 [26334,   26434,   26551,   26758,   26946,   27139,   27326 , ....]와 같이 저장되어있고, 1번행의 경우 N에 관한 R피크 위치가 [      8,     178,     354, ..., 1048192, 1048333, 1048548]와 같이 저장되어있습니다.
  <br/>
  
  각 행별 beat에 대한 label은 다음과 같습니다. `0 -> Q` , `1 -> N`, `2 -> S`, `4 -> V`

  각 행별 rythm에 대한 label은 다음과 같습니다. `0,1,2 -> noise`, `3 -> normal`, `4 -> af`, `5 -> afl`

2. PhysioNet/CinC2017
* 데이터
  * rythm기반 분석을 위한 데이터 입니다.
  * 데이터는 [CinC](https://physionet.org/content/challenge-2017/1.0.0/) 데이터를 가져왔습니다.
  * 해당 데이터에서 training2017 데이터를 사용했고, ECG-DualNet을 참고하여 `config.py`와 `util.py`와 `augmentations.py`의 클래스와 함수들을 이용해 데이터로더를 구성하였습니다.
  

