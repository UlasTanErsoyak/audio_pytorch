from torch.utils.data import Dataset
import os
import pandas as pd
import torchaudio

class Urban_Sound_8K_Dataset(Dataset):
    def __init__(self,anotations_file,audio_dir):
        self.annotations = pd.read_csv(anotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path)
        return signal,label

    def _get_audio_sample_path(self,index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        file_name = self.annotations.iloc[index,0]
        return os.path.join(self.audio_dir,fold,file_name)
    
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6]

ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
x = Urban_Sound_8K_Dataset(ANNOTATIONS_FILE,AUDIO_DIR)
print(len(x))
print(x[10])