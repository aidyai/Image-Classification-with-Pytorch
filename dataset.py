import os
import json
from PIL import Image
import pandas as pd
from sklearn import model_selection, metrics
import torch
#from torch.utils.data import Dataset 
import torch.utils.data as data

from utils import CFG, FOLD_CFG
import cv2 
from sklearn.model_selection import train_test_split, KFold



from libraries import *



### READ IN CSV FILE ### DEFINE DIRECTORIES
##########-----------##################
df = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/TRAIN.csv", encoding='unicode_escape')
 
### SPLIT FUNCTION
train_df, valid_df = train_test_split(df, test_size=CFG['test_size'])
test_df = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/TEST.csv", encoding='unicode_escape')
sample_submission = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/SampleSubmission.csv", encoding='unicode_escape')

######################

kf = KFold(n_splits=FOLD_CFG.SPLITS, shuffle=True, random_state=FOLD_CFG.SEED)




##############
####
TRAIN_PATH = "/notebooks/pixels-CLS/RICE/DATA/TRAIN-IMAGES"
TEST_PATH = "/notebooks/pixels-CLS/RICE/DATA/TEST-IMAGES"




# ====================================================
# Dataset
# ====================================================

   
    
class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['Image_id'] .values
        self.labels = df['Label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)    
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
    
    
    

class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['Image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    @staticmethod
    def collate_fn(batch):
        images = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images
